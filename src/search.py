"""
src/search.py - Path aggregation, Best-of-N search, and compute-matched search.
"""
import math
import time
from typing import Any, Callable

from .generation import (
    compute_majority_vote,
    format_step_prompt,
    resolve_generate_fn,
    run_self_consistency,
    sample_candidates,
)
from .prm_judge import score_path
from .utils import make_hf_judge_fn

GenerateFn = Callable[[str, int, dict], list[dict]]
JudgeFn = Callable[[str], dict]

AGGREGATION_STRATEGIES = ("min", "product", "step_penalty", "product_logprob")
SCORE_TIE_TOLERANCE = 1e-9


def aggregate_path_score(
    step_scores: list[float], 
    strategy: str = "min", 
    penalty_lambda: float = 2.0
) -> float:
    """
    Aggregates a list of step scores (+1.0 or -1.0) into a scalar trajectory score.
    
    Rules:
      - An empty list returns -1.0 for every strategy.
      - 'min': min(step_scores).
      - 'product': p_t = clamp((r_t + 1)/2, 0.01, 0.99); returns the raw product.
      - 'step_penalty': mean(step_scores) - penalty_lambda * count(r_t == -1.0).
      - An unknown strategy raises ValueError.
    """
    if strategy not in AGGREGATION_STRATEGIES:
        raise ValueError(
            f"Unknown aggregation strategy {strategy!r}; expected one of {AGGREGATION_STRATEGIES}"
        )

    scores = [float(r) for r in (step_scores or [])]
    if not scores:
        return -1.0  # no steps = nothing verified = worst score, for every strategy

    if strategy == "min":
        # Weakest link: one INVALID step sinks the whole path.
        return min(scores)

    if strategy == "product":
        # Map r in {-1,+1} to a validity probability p in {0.01, 0.99}; joint validity = product.
        return math.prod(min(max((r + 1.0) / 2.0, 0.01), 0.99) for r in scores)

    if strategy == "product_logprob":
        # Optional bonus. Step scores are continuous, r_t = 2 * p_t - 1, where p_t is the
        # judge's softmax probability of VALID vs INVALID (see prm_judge.score_path_logprob).
        # Same joint-validity product as 'product', but WITHOUT the 0.01/0.99 clamp, which
        # would erase the confidence information (most p_t are > 0.99).
        return math.prod(min(max((r + 1.0) / 2.0, 0.0), 1.0) for r in scores)

    # "step_penalty": average step score minus lambda per INVALID step.
    num_invalid = sum(1 for r in scores if abs(r + 1.0) < 1e-9)
    return sum(scores) / len(scores) - penalty_lambda * num_invalid


def select_best_path(
    candidate_paths: list[dict[str, Any]], 
    aggregation_strategy: str = "min", 
    penalty_lambda: float = 2.0
) -> dict[str, Any]:
    """
    Selects the winning candidate path from candidate_paths.
    
    Input Dictionary Contract:
      Each candidate dictionary in candidate_paths must contain 'step_scores' (list[float])
      and 'answer' (int | None). Any other keys present in the candidate dictionary
      are passed through unchanged in the returned copy.
    
    Selection & Tie-Breaking Rules:
      1. Compute path_score for each candidate with aggregate_path_score.
         A candidate with an empty step_scores list gets -1.0.
      2. Eligibility: if at least one candidate has a non-None answer, candidates
         with answer=None are excluded. If all answers are None, all remain eligible.
      3. Find maximum path_score among eligible candidates (tolerance 1e-9).
         The tied set is all eligible candidates within tolerance of the maximum.
      4. If the tied set has more than one candidate, compute compute_majority_vote
         over their answers. Return the lowest-index candidate in the tied set whose
         answer equals the vote. If the vote is None, return the lowest-index tied candidate.
      5. Returns a shallow COPY of the chosen dict with two added keys:
         'path_score' (float) and 'num_tied' (int, size of the tied set).
         Does not mutate the inputs.
    """
    index, path_score, num_tied = select_best_index(
        candidate_paths, aggregation_strategy=aggregation_strategy, penalty_lambda=penalty_lambda
    )
    chosen = dict(candidate_paths[index])  # shallow copy: inputs are never mutated
    chosen["path_score"] = path_score
    chosen["num_tied"] = num_tied
    return chosen


def select_best_index(
    candidate_paths: list[dict[str, Any]],
    aggregation_strategy: str = "min",
    penalty_lambda: float = 2.0,
) -> tuple[int, float, int]:
    """
    Core of select_best_path. Returns (index of the winner, its path_score, size of the
    tied set). Exposed separately so the pipeline can log WHICH candidate won.
    """
    if not candidate_paths:
        raise ValueError("select_best_path needs at least one candidate.")

    # 1. Path score per candidate (empty step_scores -> -1.0 inside aggregate_path_score).
    scores = [
        aggregate_path_score(c.get("step_scores") or [], aggregation_strategy, penalty_lambda)
        for c in candidate_paths
    ]

    # 2. Eligibility: drop answer=None candidates, unless every candidate is None.
    eligible = [i for i, c in enumerate(candidate_paths) if c.get("answer") is not None]
    if not eligible:
        eligible = list(range(len(candidate_paths)))

    # 3. Tied set = eligible candidates within tolerance of the best eligible score.
    best = max(scores[i] for i in eligible)
    tied = [i for i in eligible if abs(scores[i] - best) <= SCORE_TIE_TOLERANCE]

    # 4. Tie-break by majority vote over the tied answers; lowest index with that answer.
    winner = tied[0]
    if len(tied) > 1:
        vote = compute_majority_vote([candidate_paths[i].get("answer") for i in tied])
        if vote is not None:
            winner = next(i for i in tied if candidate_paths[i].get("answer") == vote)

    return winner, float(scores[winner]), len(tied)


def run_prm_best_of_n(
    model: Any = None, 
    judge_model: Any = None, 
    gen_tokenizer: Any = None, 
    judge_tokenizer: Any = None, 
    question: str = "", 
    n: int = 5, 
    aggregation_strategy: str = "min", 
    penalty_lambda: float = 2.0, 
    candidates: list[dict[str, Any]] | None = None, 
    generate_fn: GenerateFn | None = None, 
    judge_fn: JudgeFn | None = None
) -> dict[str, Any]:
    """
    Executes PRM-guided Best-of-N search over reasoning trajectories.
    
    If candidates is provided (shared candidate pool), does NOT sample new candidates
    (generation_calls = 0, generated_tokens = 0). Provided candidates use the
    run_self_consistency candidate format (at minimum 'steps' and 'answer').
    Otherwise, samples n candidates using run_self_consistency.
    
    Scores each candidate's steps using score_path.
    Selects the winning candidate via select_best_path.
    
    Returns:
      selected, answer, candidates [each augmented with step_scores, rationales],
      generation_calls, judge_calls, generated_tokens, judge_tokens, time_seconds
    """
    start = time.perf_counter()

    # 1. Candidate pool: reuse the shared pool if given, else sample n new paths.
    if candidates is not None:
        pool = list(candidates)
        generation_calls, generated_tokens = 0, 0
    else:
        sc = run_self_consistency(
            model=model, tokenizer=gen_tokenizer, question=question, n=n, generate_fn=generate_fn
        )
        pool = sc["candidates"]
        generation_calls, generated_tokens = sc["generation_calls"], sc["generated_tokens"]

    # 2. Judge every step of every candidate (no early stopping).
    if judge_fn is None:
        if judge_model is None or judge_tokenizer is None:
            raise ValueError("Provide either judge_fn or both judge_model and judge_tokenizer.")
        judge_fn = make_hf_judge_fn(judge_model, judge_tokenizer)

    scored: list[dict[str, Any]] = []
    judge_calls, judge_tokens = 0, 0
    for cand in pool:
        result = score_path(question=question, candidate=cand, judge_fn=judge_fn)
        augmented = dict(cand)  # copy: the caller's candidate dicts stay untouched
        augmented["step_scores"] = result["step_scores"]
        augmented["rationales"] = result["rationales"]
        scored.append(augmented)
        judge_calls += result["judge_calls"]
        judge_tokens += result["judge_tokens"]

    # 3. Filter-then-vote selection.
    selected = select_best_path(
        scored, aggregation_strategy=aggregation_strategy, penalty_lambda=penalty_lambda
    )

    return {
        "selected": selected,
        "answer": selected.get("answer"),
        "candidates": scored,
        "generation_calls": generation_calls,
        "judge_calls": judge_calls,
        "generated_tokens": generated_tokens,
        "judge_tokens": judge_tokens,
        "time_seconds": time.perf_counter() - start,
    }


def run_compute_matched_self_consistency(
    model: Any = None, 
    tokenizer: Any = None, 
    question: str = "", 
    target_calls: int = 15, 
    temperature: float = 0.7, 
    existing_candidates: list[dict[str, Any]] | None = None, 
    generate_fn: GenerateFn | None = None
) -> dict[str, Any]:
    """
    Executes compute-matched Self-Consistency aggregating target_calls paths.
    If existing_candidates are supplied, samples only the deficit:
    target_calls - len(existing_candidates). Returned candidates = existing_candidates followed by new samples, in sampling order.
    
    Returns:
      candidates, answer, generation_calls (new samples only), total_samples,
      generated_tokens (new samples only), time_seconds
    """
    start = time.perf_counter()
    existing = list(existing_candidates or [])

    # Sample only the deficit; existing candidates are reused, not regenerated.
    deficit = max(0, target_calls - len(existing))
    new: list[dict[str, Any]] = []
    if deficit > 0:
        gen = resolve_generate_fn(model, tokenizer, generate_fn)
        new = sample_candidates(gen, format_step_prompt(question), deficit, temperature)

    # Existing first, then new, in sampling order. If more existing candidates than the
    # target were passed, keep the first target_calls (same "first N samples" rule as
    # the scaling curve).
    all_candidates = (existing + new)[:max(target_calls, 0)]

    return {
        "candidates": all_candidates,
        "answer": compute_majority_vote([c.get("answer") for c in all_candidates]),
        "generation_calls": len(new),                         # new samples only
        "total_samples": len(all_candidates),                 # = target_calls
        "generated_tokens": sum(c["num_tokens"] for c in new),  # new samples only
        "time_seconds": time.perf_counter() - start,
    }
