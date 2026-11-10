"""
src/pipeline.py - End-to-end benchmark runner for Problem 4.

Runs all six configurations over the 50-problem benchmark using ONLY the functions
implemented in generation.py / prm_judge.py / search.py, checkpoints every finished
problem to outputs/per_problem.jsonl (so a dropped Kaggle session can resume), and
serializes the required artifacts:

    outputs/per_problem.jsonl          one line per problem (full traces, scores, answers)
    outputs/results_summary.json       6-row method table
    outputs/aggregation_ablation.json  min / product / step_penalty (lambda 0, 0.5, 2)
    outputs/sc_scaling_curve.json      SC accuracy for N = 1..15

Per problem (seed = 5326 + problem_index, set before the problem's sampling):
    1. Greedy (N=1)                       run_greedy
    2. 5 shared samples -> SC@5           run_self_consistency
    3. Judge the 5 shared candidates      (batched prefetch into a cache)
    4. PRM Best-of-5 x {min, product, step_penalty}  run_prm_best_of_n on the SAME 5
    5. Aggregation ablation               select_best_index on the cached step scores
    6. 10 extra samples -> SC@15          run_compute_matched_self_consistency
"""
import json
import os
import re
import time
from typing import Any, Callable, Iterable

from .generation import compute_majority_vote, run_greedy, run_self_consistency
from .prm_judge import build_prm_judge_prompt
from .search import (
    run_compute_matched_self_consistency,
    run_prm_best_of_n,
    select_best_index,
)
from .utils import (
    save_aggregation_ablation,
    save_results_summary,
    save_sc_scaling_curve,
    set_seed_for_problem,
)

# ---------------------------------------------------------------------------
# Constants (method names must match the manual's results table exactly)
# ---------------------------------------------------------------------------
GREEDY = "Greedy (N=1)"
SC5 = "Self-Consistency (N=5)"
SC15 = "Compute-Matched SC (N=15)"
PRM_METHODS = [  # (method name, aggregation strategy, lambda)
    ("PRM Best-of-5 (min)", "min", 2.0),
    ("PRM Best-of-5 (product)", "product", 2.0),
    ("PRM Best-of-5 (step_penalty, lambda=2.0)", "step_penalty", 2.0),
]
METHOD_ORDER = [GREEDY, SC5, SC15] + [m for m, _, _ in PRM_METHODS]

# (strategy, lambda) rows of aggregation_ablation.json; lambda is None for min/product.
ABLATION_CONFIGS = [
    ("min", None),
    ("product", None),
    ("step_penalty", 0.0),
    ("step_penalty", 0.5),
    ("step_penalty", 2.0),
]

N_SHARED = 5
SC_TARGET = 15
TEMPERATURE = 0.7

_VERDICT_LINE_RE = re.compile(r"verdict\s*:\s*\**\s*(VALID|INVALID)\b", re.IGNORECASE)


# ---------------------------------------------------------------------------
# JSONL helpers
# ---------------------------------------------------------------------------
def read_jsonl(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass  # a half-written last line from a killed session: drop it
    return rows


def append_jsonl(path: str, row: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())  # survive an abrupt session kill


def write_jsonl(path: str, rows: Iterable[dict]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def load_benchmark(path: str = "data/gsm8k_test_50.jsonl") -> list[dict]:
    return read_jsonl(path)


def is_correct(answer: Any, ground_truth: Any) -> bool:
    return answer is not None and int(answer) == int(ground_truth)


# ---------------------------------------------------------------------------
# Judge cache: batched prefetch + JudgeFn interface
# ---------------------------------------------------------------------------
class CachedJudge:
    """
    A JudgeFn (prompt -> {"text", "num_tokens"}) backed by a prompt->result cache.

    prefetch(prompts) judges all not-yet-seen prompts in GPU batches (the manual:
    "Batch judge prompts across steps and candidates where possible"). Afterwards
    score_path / run_prm_best_of_n call this object once per step exactly as with an
    unbatched judge, so the bookkeeping (one judge call per step) is unchanged, and the
    three PRM aggregation rows reuse the same verdicts instead of re-running the judge.
    """

    def __init__(
        self,
        judge_fn: Callable[[str], dict] | None = None,
        batch_judge_fn: Callable[[list[str]], list[dict]] | None = None,
        batch_size: int = 16,
    ):
        if judge_fn is None and batch_judge_fn is None:
            raise ValueError("CachedJudge needs judge_fn or batch_judge_fn.")
        self.judge_fn = judge_fn
        self.batch_judge_fn = batch_judge_fn
        self.batch_size = max(1, int(batch_size))
        self.cache: dict[str, dict] = {}

    def __call__(self, prompt: str) -> dict:
        if prompt not in self.cache:
            if self.judge_fn is not None:
                self.cache[prompt] = self.judge_fn(prompt)
            else:
                self.cache[prompt] = self.batch_judge_fn([prompt])[0]
        return self.cache[prompt]

    def prefetch(self, prompts: list[str]) -> int:
        """Judges every uncached prompt; returns how many were actually computed."""
        todo = list(dict.fromkeys(p for p in prompts if p not in self.cache))
        if self.batch_judge_fn is None:
            for p in todo:
                self(p)
            return len(todo)
        # Similar lengths in the same batch -> less padding waste.
        todo.sort(key=len)
        for i in range(0, len(todo), self.batch_size):
            self._judge_batch(todo[i:i + self.batch_size])
        return len(todo)

    def _judge_batch(self, batch: list[str]) -> None:
        try:
            outputs = self.batch_judge_fn(batch)
        except Exception as exc:  # CUDA OOM -> split the batch in half and retry
            if "out of memory" not in str(exc).lower() or len(batch) == 1:
                raise
            _empty_cuda_cache()
            mid = len(batch) // 2
            self._judge_batch(batch[:mid])
            self._judge_batch(batch[mid:])
            return
        for p, out in zip(batch, outputs):
            self.cache[p] = out

    def clear(self) -> None:
        self.cache.clear()


def _empty_cuda_cache() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def judge_prompts_for(question: str, candidates: list[dict]) -> list[str]:
    """Exactly the prompts score_path will build: step t judged with steps[:t] as context."""
    prompts = []
    for cand in candidates:
        steps = cand.get("steps") or []
        for t, step in enumerate(steps):
            prompts.append(build_prm_judge_prompt(question, steps[:t], step))
    return prompts


# ---------------------------------------------------------------------------
# Phase A: greedy over all problems (+ format-compliance check)
# ---------------------------------------------------------------------------
def greedy_record(problem_index: int, row: dict, greedy: dict) -> dict:
    return {
        "problem_index": problem_index,
        "id": row["id"],
        "greedy": {
            "completion": greedy["completion"],
            "steps": greedy["steps"],
            "answer": greedy["answer"],
            "correct": is_correct(greedy["answer"], row["ground_truth_answer"]),
            "generation_calls": greedy["generation_calls"],
            "generated_tokens": greedy["generated_tokens"],
            "time_seconds": greedy["time_seconds"],
        },
    }


def run_greedy_phase(
    rows: list[dict],
    generate_fn: Callable,
    checkpoint_path: str = "outputs/greedy_checkpoint.jsonl",
    progress: Callable | None = None,
) -> dict[str, dict]:
    """Runs greedy on every problem not already in the checkpoint. Returns {id: record}."""
    done = {r["id"]: r for r in read_jsonl(checkpoint_path)}
    todo = [(i, row) for i, row in enumerate(rows) if row["id"] not in done]
    iterator = progress(todo, desc="Greedy") if progress else todo
    for i, row in iterator:
        set_seed_for_problem(i)
        greedy = run_greedy(question=row["question"], generate_fn=generate_fn)
        rec = greedy_record(i, row, greedy)
        append_jsonl(checkpoint_path, rec)
        done[row["id"]] = rec
    return done


_DRIFT_MARKERS = {
    "boxed": re.compile(r"\\boxed"),
    "latex": re.compile(r"\\\(|\\\[|\$\$|\\frac|\\times|\\cdot"),
    "markdown_heading": re.compile(r"(?m)^\s*#{1,6}\s"),
    "bold": re.compile(r"\*\*"),
}


def compliance_report(greedy_records: dict[str, dict], rows: list[dict]) -> dict:
    """Delimiter / format compliance of greedy outputs (manual 4.2 warning)."""
    recs = [greedy_records[r["id"]]["greedy"] for r in rows if r["id"] in greedy_records]
    n = len(recs)
    if n == 0:
        return {"num_problems": 0}
    has_delim = sum(bool(re.search(r"final\s*answer", g["completion"], re.I)) for g in recs)
    parsed = sum(g["answer"] is not None for g in recs)
    has_steps = sum(len(g["steps"]) > 0 for g in recs)
    drift = {k: sum(bool(rx.search(g["completion"])) for g in recs) for k, rx in _DRIFT_MARKERS.items()}
    failures = [
        r["id"] for r in rows
        if r["id"] in greedy_records and greedy_records[r["id"]]["greedy"]["answer"] is None
    ]
    return {
        "num_problems": n,
        "final_answer_delimiter_rate_%": round(100 * has_delim / n, 2),
        "answer_parsed_rate_%": round(100 * parsed / n, 2),
        "has_steps_rate_%": round(100 * has_steps / n, 2),
        "mean_steps": round(sum(len(g["steps"]) for g in recs) / n, 2),
        "greedy_accuracy_%": round(100 * sum(g["correct"] for g in recs) / n, 2),
        "drift_counts": drift,
        "unparsed_ids": failures,
    }


# ---------------------------------------------------------------------------
# Phase B: everything else, one problem at a time
# ---------------------------------------------------------------------------
def run_problem(
    problem_index: int,
    row: dict,
    generate_fn: Callable,
    judge: CachedJudge,
    greedy_rec: dict | None = None,
) -> dict:
    """Runs all six configurations (+ ablation + curve) for one problem; returns its record."""
    q, gt = row["question"], row["ground_truth_answer"]

    # 1. Greedy (normally precomputed in phase A; greedy decoding uses no RNG).
    if greedy_rec is None:
        set_seed_for_problem(problem_index)
        greedy_rec = greedy_record(problem_index, row, run_greedy(question=q, generate_fn=generate_fn))
    greedy = greedy_rec["greedy"]

    # 2. Shared pool of 5 samples -> SC@5. Seed set right before the problem's sampling.
    set_seed_for_problem(problem_index)
    sc5 = run_self_consistency(question=q, n=N_SHARED, temperature=TEMPERATURE, generate_fn=generate_fn)
    shared = sc5["candidates"]

    # 3. Judge every step of the 5 shared candidates ONCE (batched, cached).
    prompts = judge_prompts_for(q, shared)
    t0 = time.perf_counter()
    unique_judged = judge.prefetch(prompts)
    prefetch_time = time.perf_counter() - t0

    # 4. PRM Best-of-5, one row per aggregation rule, all on the same 5 candidates.
    #    Verdicts come from the cache, so each row reports identical judge calls/tokens.
    prm = {}
    for method, strategy, lam in PRM_METHODS:
        prm[method] = run_prm_best_of_n(
            question=q, n=N_SHARED, aggregation_strategy=strategy, penalty_lambda=lam,
            candidates=shared, judge_fn=judge,
        )
    scored = prm[PRM_METHODS[0][0]]["candidates"]  # step_scores identical across rows
    judge_calls = prm[PRM_METHODS[0][0]]["judge_calls"]
    judge_tokens = prm[PRM_METHODS[0][0]]["judge_tokens"]

    # 5. Aggregation ablation on the cached step scores (no extra judging).
    ablation = []
    for strategy, lam in ABLATION_CONFIGS:
        idx, score, tied = select_best_index(scored, strategy, 2.0 if lam is None else lam)
        ans = scored[idx]["answer"]
        ablation.append({
            "strategy": strategy, "lambda": lam, "answer": ans, "correct": is_correct(ans, gt),
            "selected_index": idx, "path_score": score, "num_tied": tied,
        })

    # 6. Compute-matched SC@15: keep the 5 shared samples, draw 10 more.
    sc15 = run_compute_matched_self_consistency(
        question=q, target_calls=SC_TARGET, temperature=TEMPERATURE,
        existing_candidates=shared, generate_fn=generate_fn,
    )
    all15 = sc15["candidates"]

    # --- assemble the record ---
    candidates = []
    for i, c in enumerate(all15):
        entry = {
            "index": i,
            "pool": "shared" if i < N_SHARED else "sc15_extra",
            "completion": c["completion"],
            "steps": c["steps"],
            "answer": c["answer"],
            "num_tokens": c["num_tokens"],
        }
        if i < N_SHARED:
            steps = c["steps"]
            entry["step_scores"] = scored[i]["step_scores"]
            entry["rationales"] = scored[i]["rationales"]
            entry["judge_outputs"] = [
                judge(build_prm_judge_prompt(q, steps[:t], steps[t]))["text"]
                for t in range(len(steps))
            ]
        candidates.append(entry)

    judge_texts = [o for c in candidates[:N_SHARED] for o in c["judge_outputs"]]
    methods = {
        GREEDY: {"answer": greedy["answer"], "correct": greedy["correct"]},
        SC5: {"answer": sc5["answer"], "correct": is_correct(sc5["answer"], gt)},
        SC15: {"answer": sc15["answer"], "correct": is_correct(sc15["answer"], gt)},
    }
    for method, strategy, lam in PRM_METHODS:
        idx, score, tied = select_best_index(scored, strategy, lam)
        ans = prm[method]["answer"]
        assert ans == scored[idx]["answer"]  # same selection rule, same result
        methods[method] = {
            "answer": ans, "correct": is_correct(ans, gt),
            "selected_index": idx, "path_score": score, "num_tied": tied,
        }

    curve = []
    for n in range(1, SC_TARGET + 1):
        ans = compute_majority_vote([c["answer"] for c in all15[:n]])
        curve.append({
            "N": n, "answer": ans, "correct": is_correct(ans, gt),
            "gen_tokens": sum(c["num_tokens"] for c in all15[:n]),
        })

    return {
        "problem_index": problem_index,
        "id": row["id"],
        "question": q,
        "ground_truth_answer": gt,
        "greedy": greedy,
        "candidates": candidates,
        "methods": methods,
        "ablation": ablation,
        "sc_curve": curve,
        "costs": {
            "greedy": {k: greedy[k] for k in ("generation_calls", "generated_tokens", "time_seconds")},
            "sc5": {k: sc5[k] for k in ("generation_calls", "generated_tokens", "time_seconds")},
            "sc15_extra": {k: sc15[k] for k in ("generation_calls", "generated_tokens", "time_seconds")},
            "judge": {
                "judge_calls": judge_calls,
                "judge_tokens": judge_tokens,
                "unique_prompts_judged": unique_judged,
                "prefetch_time_seconds": prefetch_time,
            },
            # Each PRM row's own run_prm_best_of_n time (cache lookups + its selection).
            "prm_call_time_seconds": {m: prm[m]["time_seconds"] for m, _, _ in PRM_METHODS},
        },
        "judge_format": {
            "steps_judged": len(judge_texts),
            "verdict_line_missing": sum(not _VERDICT_LINE_RE.search(t) for t in judge_texts),
        },
    }


def run_main_phase(
    rows: list[dict],
    generate_fn: Callable,
    judge: CachedJudge,
    greedy_records: dict[str, dict] | None = None,
    out_path: str = "outputs/per_problem.jsonl",
    progress: Callable | None = None,
    log: Callable[[str], None] = print,
) -> list[dict]:
    """Runs every problem not yet in out_path, appending each finished record immediately."""
    greedy_records = greedy_records or {}
    done = {r["id"]: r for r in read_jsonl(out_path)}
    todo = [(i, row) for i, row in enumerate(rows) if row["id"] not in done]
    if done:
        log(f"Resuming: {len(done)} problems already done, {len(todo)} to go.")
    iterator = progress(todo, desc="Main run") if progress else todo
    for i, row in iterator:
        rec = run_problem(i, row, generate_fn, judge, greedy_records.get(row["id"]))
        append_jsonl(out_path, rec)
        done[row["id"]] = rec
        judge.clear()  # prompts never repeat across problems; free the memory
        m = rec["methods"]
        log(
            f"[{rec['id']}] gt={rec['ground_truth_answer']} | greedy={m[GREEDY]['answer']} "
            f"sc5={m[SC5]['answer']} sc15={m[SC15]['answer']} "
            f"prm_min={m[PRM_METHODS[0][0]]['answer']} | judge_calls={rec['costs']['judge']['judge_calls']}"
        )
    return [done[r["id"]] for r in rows if r["id"] in done]


# ---------------------------------------------------------------------------
# Artifacts
# ---------------------------------------------------------------------------
def _pct(xs: list[bool]) -> float:
    return round(100.0 * sum(xs) / len(xs), 2) if xs else 0.0


def build_artifacts(records: list[dict], out_dir: str = "outputs") -> dict[str, Any]:
    """Computes and writes results_summary / aggregation_ablation / sc_scaling_curve,
    and rewrites per_problem.jsonl sorted by problem index."""
    if not records:
        raise ValueError("No per-problem records to summarize.")
    records = sorted(records, key=lambda r: r["problem_index"])
    n = len(records)

    def total(path: tuple[str, ...]) -> float:
        out = 0.0
        for r in records:
            v: Any = r["costs"]
            for k in path:
                v = v[k]
            out += v
        return out

    def acc(method: str) -> float:
        return _pct([r["methods"][method]["correct"] for r in records])

    def row(method, gen_calls, judge_calls, gen_tok, judge_tok, secs):
        return {
            "Method": method,
            "Accuracy (%)": acc(method),
            "Generation Calls": int(gen_calls),
            "Judge Calls": int(judge_calls),
            "Total LLM Calls": int(gen_calls + judge_calls),
            "Avg Gen Tokens / Problem": round(gen_tok / n, 2),
            "Avg Judge Tokens / Problem": round(judge_tok / n, 2),
            "Total Time (s)": round(secs, 2),
        }

    g_calls, g_tok, g_time = (total(("greedy", k)) for k in ("generation_calls", "generated_tokens", "time_seconds"))
    s5_calls, s5_tok, s5_time = (total(("sc5", k)) for k in ("generation_calls", "generated_tokens", "time_seconds"))
    ex_calls, ex_tok, ex_time = (total(("sc15_extra", k)) for k in ("generation_calls", "generated_tokens", "time_seconds"))
    j_calls, j_tok, j_time = (total(("judge", k)) for k in ("judge_calls", "judge_tokens", "prefetch_time_seconds"))

    # Time attribution (manual 4.1): Greedy = own time; SC@5 = sampling the 5 shared;
    # SC@15 = SC@5 time + the 10 extra; each PRM row = SC@5 time + judging + its own selection.
    summary = [
        row(GREEDY, g_calls, 0, g_tok, 0, g_time),
        row(SC5, s5_calls, 0, s5_tok, 0, s5_time),
        row(SC15, s5_calls + ex_calls, 0, s5_tok + ex_tok, 0, s5_time + ex_time),
    ]
    for method, _, _ in PRM_METHODS:
        own = total(("prm_call_time_seconds", method))
        summary.append(row(method, s5_calls, j_calls, s5_tok, j_tok, s5_time + j_time + own))

    ablation = []
    for k, (strategy, lam) in enumerate(ABLATION_CONFIGS):
        ablation.append({
            "Strategy": strategy,
            "Lambda": lam,
            "Accuracy (%)": _pct([r["ablation"][k]["correct"] for r in records]),
            "Tie Rate (%)": _pct([r["ablation"][k]["num_tied"] > 1 for r in records]),
        })

    curve = []
    for k in range(SC_TARGET):
        curve.append({
            "N": k + 1,
            "Accuracy (%)": _pct([r["sc_curve"][k]["correct"] for r in records]),
            "Avg Gen Tokens / Problem": round(sum(r["sc_curve"][k]["gen_tokens"] for r in records) / n, 2),
        })

    # Consistency checks required by the manual.
    by_method = {s["Method"]: s for s in summary}
    assert curve[N_SHARED - 1]["Accuracy (%)"] == by_method[SC5]["Accuracy (%)"], "curve N=5 != SC@5"
    assert curve[SC_TARGET - 1]["Accuracy (%)"] == by_method[SC15]["Accuracy (%)"], "curve N=15 != SC@15"
    for method, strategy, lam in PRM_METHODS:
        k = next(i for i, (s, l) in enumerate(ABLATION_CONFIGS) if s == strategy and (l is None or l == lam))
        assert ablation[k]["Accuracy (%)"] == by_method[method]["Accuracy (%)"], f"ablation != {method}"

    os.makedirs(out_dir, exist_ok=True)
    save_results_summary(summary, os.path.join(out_dir, "results_summary.json"))
    save_aggregation_ablation(ablation, os.path.join(out_dir, "aggregation_ablation.json"))
    save_sc_scaling_curve(curve, os.path.join(out_dir, "sc_scaling_curve.json"))
    write_jsonl(os.path.join(out_dir, "per_problem.jsonl"), records)

    judge_missing = sum(r["judge_format"]["verdict_line_missing"] for r in records)
    judged = sum(r["judge_format"]["steps_judged"] for r in records)
    return {
        "num_problems": n,
        "summary": summary,
        "ablation": ablation,
        "curve": curve,
        "judge_verdict_line_missing": f"{judge_missing}/{judged}",
    }


# ---------------------------------------------------------------------------
# Optional bonus: continuous logprob PRM ("product_logprob")
# ---------------------------------------------------------------------------
# (variant name, which per-step probabilities it uses)
LOGPROB_VARIANTS = [
    ("product_logprob", "step_probs"),                              # full-sequence (main)
    ("product_logprob_first_token", "step_probs_first_token"),      # first-token comparison
]


def _call_with_oom_split(fn: Callable[[list], list], batch: list) -> list:
    try:
        return fn(batch)
    except Exception as exc:  # CUDA OOM -> halve the batch and retry
        if "out of memory" not in str(exc).lower() or len(batch) == 1:
            raise
        _empty_cuda_cache()
        mid = len(batch) // 2
        return _call_with_oom_split(fn, batch[:mid]) + _call_with_oom_split(fn, batch[mid:])


class CachedLogprob:
    """LogprobFn (list of judge prompts -> list of logprob dicts) with batched prefetch + cache."""

    def __init__(self, batch_logprob_fn: Callable[[list[str]], list[dict]], batch_size: int = 16):
        self.fn = batch_logprob_fn
        self.batch_size = max(1, int(batch_size))
        self.cache: dict[str, dict] = {}

    def prefetch(self, prompts: list[str]) -> int:
        todo = sorted(dict.fromkeys(p for p in prompts if p not in self.cache), key=len)
        for i in range(0, len(todo), self.batch_size):
            chunk = todo[i:i + self.batch_size]
            for p, out in zip(chunk, _call_with_oom_split(self.fn, chunk)):
                self.cache[p] = out
        return len(todo)

    def __call__(self, prompts: list[str]) -> list[dict]:
        self.prefetch(prompts)
        return [self.cache[p] for p in prompts]

    def clear(self) -> None:
        self.cache.clear()


def logprob_problem(record: dict, scorer: CachedLogprob) -> dict:
    """Rescores the 5 shared candidates of one main-run record with continuous step
    probabilities and selects with 'product_logprob' (filter-then-vote, as before)."""
    from .prm_judge import score_path_logprob  # local import keeps the main path lean

    q, gt = record["question"], record["ground_truth_answer"]
    shared = record["candidates"][:N_SHARED]

    t0 = time.perf_counter()
    scorer.prefetch(judge_prompts_for(q, shared))  # all steps of the problem, batched
    cands = []
    for c in shared:
        res = score_path_logprob(question=q, candidate=c, logprob_fn=scorer)
        cands.append({
            "index": c["index"],
            "answer": c["answer"],
            "step_scores_binary": c["step_scores"],
            "step_probs": res["step_probs"],
            "step_probs_first_token": res["step_probs_first_token"],
        })
    selections = {}
    for variant, key in LOGPROB_VARIANTS:
        scored = [{"answer": c["answer"], "step_scores": [2.0 * p - 1.0 for p in c[key]]} for c in cands]
        idx, score, tied = select_best_index(scored, "product_logprob")
        ans = cands[idx]["answer"]
        selections[variant] = {
            "answer": ans, "correct": is_correct(ans, gt),
            "selected_index": idx, "path_score": score, "num_tied": tied,
        }
    elapsed = time.perf_counter() - t0
    scorer.clear()
    return {
        "problem_index": record["problem_index"],
        "id": record["id"],
        "ground_truth_answer": gt,
        "candidates": cands,
        "selections": selections,
        "cost": {"logprob_calls": sum(len(c["step_probs"]) for c in cands), "time_seconds": elapsed},
    }


def run_logprob_phase(
    main_records: list[dict],
    scorer: CachedLogprob,
    out_path: str = "outputs/logprob_bonus.jsonl",
    progress: Callable | None = None,
) -> list[dict]:
    done = {r["id"]: r for r in read_jsonl(out_path)}
    todo = [r for r in sorted(main_records, key=lambda r: r["problem_index"]) if r["id"] not in done]
    iterator = progress(todo, desc="Logprob bonus") if progress else todo
    for rec in iterator:
        bonus = logprob_problem(rec, scorer)
        append_jsonl(out_path, bonus)
        done[rec["id"]] = bonus
    return [done[r["id"]] for r in sorted(main_records, key=lambda r: r["problem_index"]) if r["id"] in done]


def build_logprob_artifacts(
    bonus_records: list[dict],
    main_records: list[dict],
    out_dir: str = "outputs",
    approach: str = "",
) -> dict[str, Any]:
    """Writes outputs/logprob_bonus_summary.json (bonus results + comparison with the
    discrete-verdict PRM) and rewrites outputs/logprob_bonus.jsonl sorted."""
    bonus_records = sorted(bonus_records, key=lambda r: r["problem_index"])
    main = {r["id"]: r for r in main_records}
    n = len(bonus_records)

    rows = []
    for variant, _ in LOGPROB_VARIANTS:
        rows.append({
            "Strategy": variant,
            "Accuracy (%)": _pct([b["selections"][variant]["correct"] for b in bonus_records]),
            "Tie Rate (%)": _pct([b["selections"][variant]["num_tied"] > 1 for b in bonus_records]),
        })
    # Discrete-verdict references on the same problems (from the main run)
    for k, (strategy, lam) in enumerate(ABLATION_CONFIGS):
        rows.append({
            "Strategy": f"{strategy} (binary)" if lam is None else f"{strategy} lambda={lam} (binary)",
            "Accuracy (%)": _pct([main[b["id"]]["ablation"][k]["correct"] for b in bonus_records]),
            "Tie Rate (%)": _pct([main[b["id"]]["ablation"][k]["num_tied"] > 1 for b in bonus_records]),
        })

    steps = [(s, p, pf) for b in bonus_records for c in b["candidates"]
             for s, p, pf in zip(c["step_scores_binary"], c["step_probs"], c["step_probs_first_token"])]
    p_valid = [p for s, p, _ in steps if s > 0]
    p_invalid = [p for s, p, _ in steps if s < 0]
    product_k = ABLATION_CONFIGS.index(("product", None))
    changed = sum(
        b["selections"]["product_logprob"]["selected_index"] != main[b["id"]]["ablation"][product_k]["selected_index"]
        for b in bonus_records
    )
    summary = {
        "approach": approach,
        "num_problems": n,
        "rows": rows,
        "steps_scored": len(steps),
        "verdict_agreement_%": _pct([(p > 0.5) == (s > 0) for s, p, _ in steps]),
        "mean_p_valid_when_judge_said_VALID": round(sum(p_valid) / len(p_valid), 6) if p_valid else None,
        "mean_p_valid_when_judge_said_INVALID": round(sum(p_invalid) / len(p_invalid), 6) if p_invalid else None,
        "steps_with_p_between_0.1_and_0.9": sum(0.1 < p < 0.9 for _, p, _ in steps),
        "max_abs_diff_full_vs_first_token": max((abs(p - pf) for _, p, pf in steps), default=0.0),
        "problems_where_selection_differs_from_binary_product": changed,
        "total_logprob_calls": sum(b["cost"]["logprob_calls"] for b in bonus_records),
        "total_time_s": round(sum(b["cost"]["time_seconds"] for b in bonus_records), 2),
    }
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "logprob_bonus_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    write_jsonl(os.path.join(out_dir, "logprob_bonus.jsonl"), bonus_records)
    return summary


def format_table(rows: list[dict]) -> str:
    """Plain-text table for notebook output (no extra dependencies)."""
    if not rows:
        return ""
    cols = list(rows[0])
    cells = [[str(r[c]) for c in cols] for r in rows]
    widths = [max(len(c), *(len(x[i]) for x in cells)) for i, c in enumerate(cols)]
    line = lambda xs: " | ".join(x.ljust(w) for x, w in zip(xs, widths))
    return "\n".join([line(cols), "-+-".join("-" * w for w in widths)] + [line(x) for x in cells])
