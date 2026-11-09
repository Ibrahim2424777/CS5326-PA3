"""
src/generation.py - Prompt formatting and baseline generation routines.
"""
import time
from typing import Any, Callable

from .extraction import extract_answer, parse_steps
from .utils import GREEDY_GEN_KWARGS, SAMPLING_GEN_KWARGS, make_hf_generate_fn

GenerateFn = Callable[[str, int, dict], list[dict]]
# generate_fn(raw_prompt, num_samples, gen_kwargs) -> list of {"text": str, "num_tokens": int}
# gen_kwargs carries do_sample/temperature/top_p/top_k/repetition_penalty/max_new_tokens


def format_step_prompt(question: str) -> str:
    """
    Formats the raw problem question into a prompt that instructs the 
    model to emit numbered steps ('Step 1: ...') and 'Final Answer: <int>'.
    
    Returns a raw instruction string (no chat template applied).
    """
    return STEP_PROMPT_TEMPLATE.format(question=question.strip())


# Zero-shot instruction. The format block shows the schema with placeholders only
# (no worked example), so this stays zero-shot. The rules target the failure modes
# the manual reports for Qwen2.5-1.5B: markdown headings, LaTeX, \boxed{}, and a
# missing "Final Answer:" line.
STEP_PROMPT_TEMPLATE = (
    "Solve the following math word problem by reasoning step by step.\n"
    "\n"
    "Problem: {question}\n"
    "\n"
    "Write your solution in exactly this format:\n"
    "Step 1: <first deduction or calculation>\n"
    "Step 2: <next deduction or calculation>\n"
    "...\n"
    "Step K: <final calculation>\n"
    "Final Answer: <integer>\n"
    "\n"
    "Rules:\n"
    "- Begin every step on a new line with \"Step N:\" where N counts up from 1.\n"
    "- Each step makes one deduction and writes out its arithmetic, e.g. \"3 + 4 = 7\".\n"
    "- Use plain text only. Do not use markdown (no #, no **, no bullet lists), "
    "do not use LaTeX, and do not use \\boxed{{}}.\n"
    "- The last line must be \"Final Answer: \" followed by the answer as a whole number "
    "written with digits only (no units, no $ sign, no commas).\n"
    "- Do not write anything after the Final Answer line."
)


def compute_majority_vote(answers: list[int | None]) -> int | None:
    """
    Given a list of extracted answers, returns the majority answer.
    Ignores None values. Breaks ties deterministically in favor of the answer
    that appeared earliest in the list.
    """
    counts: dict[int, int] = {}
    first_seen: dict[int, int] = {}
    for idx, ans in enumerate(answers or []):
        if ans is None:
            continue
        counts[ans] = counts.get(ans, 0) + 1
        first_seen.setdefault(ans, idx)

    if not counts:
        return None  # empty list or all None

    # Highest count wins; among equal counts, the earliest first appearance wins.
    return min(counts, key=lambda a: (-counts[a], first_seen[a]))


# ---------------------------------------------------------------------------
# Shared helpers (also used by src/search.py for compute-matched SC)
# ---------------------------------------------------------------------------
def resolve_generate_fn(model: Any, tokenizer: Any, generate_fn: GenerateFn | None) -> GenerateFn:
    """Uses the injected generate_fn (mocks / autograder) or builds the real HF one."""
    if generate_fn is not None:
        return generate_fn
    if model is None or tokenizer is None:
        raise ValueError("Provide either generate_fn or both model and tokenizer.")
    return make_hf_generate_fn(model, tokenizer)


def build_candidate(output: dict) -> dict[str, Any]:
    """Turns one generate_fn output {"text", "num_tokens"} into a candidate record."""
    text = output.get("text", "") or ""
    return {
        "completion": text,
        "steps": parse_steps(text),
        "answer": extract_answer(text),
        "num_tokens": int(output.get("num_tokens", 0)),
    }


def sample_candidates(
    generate_fn: GenerateFn, prompt: str, n: int, temperature: float
) -> list[dict[str, Any]]:
    """Draws n temperature samples in ONE batched generate_fn call (num_samples=n)."""
    if n <= 0:
        return []
    gen_kwargs = dict(SAMPLING_GEN_KWARGS)  # copy: never mutate the shared constant
    gen_kwargs["temperature"] = temperature
    outputs = generate_fn(prompt, n, gen_kwargs)
    return [build_candidate(out) for out in outputs]


def run_greedy(
    model: Any = None, 
    tokenizer: Any = None, 
    question: str = "", 
    generate_fn: GenerateFn | None = None
) -> dict[str, Any]:
    """
    Executes greedy decoding for a single question.
    Calls generate_fn with num_samples=1 and gen_kwargs containing do_sample=False,
    temperature=None, top_p=None, top_k=None.
    
    Returns:
      prompt, completion, steps, answer, generation_calls, generated_tokens, time_seconds
    """
    start = time.perf_counter()
    gen = resolve_generate_fn(model, tokenizer, generate_fn)
    prompt = format_step_prompt(question)

    # GREEDY_GEN_KWARGS = do_sample=False, temperature/top_p/top_k=None, max_new_tokens=512
    outputs = gen(prompt, 1, dict(GREEDY_GEN_KWARGS))
    cand = build_candidate(outputs[0])

    return {
        "prompt": prompt,
        "completion": cand["completion"],
        "steps": cand["steps"],
        "answer": cand["answer"],
        "generation_calls": len(outputs),  # one generated sequence = one call
        "generated_tokens": cand["num_tokens"],
        "time_seconds": time.perf_counter() - start,
    }


def run_self_consistency(
    model: Any = None, 
    tokenizer: Any = None, 
    question: str = "", 
    n: int = 5, 
    temperature: float = 0.7, 
    generate_fn: GenerateFn | None = None
) -> dict[str, Any]:
    """
    Samples n diverse reasoning trajectories and aggregates candidate answers via majority vote.
    
    Returns:
      candidates [each: completion, steps, answer, num_tokens], answer, 
      generation_calls, generated_tokens, time_seconds
    """
    start = time.perf_counter()
    gen = resolve_generate_fn(model, tokenizer, generate_fn)
    prompt = format_step_prompt(question)

    candidates = sample_candidates(gen, prompt, n, temperature)

    return {
        "candidates": candidates,
        "answer": compute_majority_vote([c["answer"] for c in candidates]),
        "generation_calls": len(candidates),  # one generated sequence = one call
        "generated_tokens": sum(c["num_tokens"] for c in candidates),
        "time_seconds": time.perf_counter() - start,
    }
