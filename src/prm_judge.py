"""
src/prm_judge.py - PRM judge prompt construction, parsing, and step scoring.
"""
import re
from typing import Any, Callable

from .utils import make_hf_judge_fn

JudgeFn = Callable[[str], dict]
# judge_fn(judge_prompt) -> {"text": str, "num_tokens": int}


# Judge prompt. Contains every literal the autograder checks:
#   "Step i: <text>" for preceding steps, "(no preceding steps)" when empty,
#   "Current Step:", "assume all preceding steps are correct",
#   "Verdict: VALID", "Verdict: INVALID", "Rationale:".
# Verdict comes first so the decision is the first thing generated (cheap to parse,
# and the verdict token's logits are available directly for the optional logprob bonus).
JUDGE_PROMPT_TEMPLATE = (
    "You are a careful math teacher checking ONE step of a student's step-by-step "
    "solution to a math word problem.\n"
    "\n"
    "Problem:\n"
    "{question}\n"
    "\n"
    "Preceding steps:\n"
    "{preceding}\n"
    "\n"
    "Current Step: {current_step}\n"
    "\n"
    "Decide whether the Current Step is VALID or INVALID.\n"
    "Assume all preceding steps are correct, even if they contain mistakes, and judge "
    "only the Current Step.\n"
    "- VALID: every calculation in the Current Step is arithmetically correct, and it "
    "follows logically from the problem statement and the preceding steps as written.\n"
    "- INVALID: it contains an arithmetic error, uses a number or fact that is not in the "
    "problem or the preceding steps, misreads what the problem says, or draws a conclusion "
    "that does not follow.\n"
    "Do not mark the Current Step INVALID only because an earlier step is wrong: if it "
    "correctly uses the results of the preceding steps, it is VALID.\n"
    "Check each calculation in the Current Step carefully before deciding.\n"
    "\n"
    "Answer with exactly two lines and nothing else.\n"
    "Line 1 must be either \"Verdict: VALID\" or \"Verdict: INVALID\".\n"
    "Line 2 must be \"Rationale:\" followed by one sentence explaining your decision."
)

NO_PRECEDING_STEPS = "(no preceding steps)"


# --- Verdict parsing patterns (manual Deliverable 2.3) ---
# Rule 1: explicit verdict line, e.g. "Verdict: VALID", "Verdict: **INVALID**".
_VERDICT_LINE_RE = re.compile(r"verdict\s*:\s*\**\s*(VALID|INVALID)\b", re.IGNORECASE)
# Rule 2: standalone INVALID or the phrase NOT VALID. Word boundaries mean the
# "VALID" inside "INVALID" can never be mistaken for a VALID verdict.
_NEGATIVE_RE = re.compile(r"\bINVALID\b|\bNOT\s+VALID\b", re.IGNORECASE)
# Rule 3: standalone VALID.
_POSITIVE_RE = re.compile(r"\bVALID\b", re.IGNORECASE)
# Rationale label, and a sentence end: . ! ? followed by whitespace or end of string
# (so the "." in "2.5" is not a sentence end).
_RATIONALE_LABEL_RE = re.compile(r"rationale\s*:", re.IGNORECASE)
_SENTENCE_END_RE = re.compile(r"[.!?](?=\s|$)")


def build_prm_judge_prompt(question: str, preceding_steps: list[str], current_step: str) -> str:
    """
    Builds the verification prompt for the PRM judge to classify current_step
    as VALID or INVALID given the problem question and preceding_steps.
    
    Judge Semantics (Local Validity):
    The judge evaluates current_step ASSUMING ALL PRECEDING STEPS ARE CORRECT.
    A step is VALID if its arithmetic is sound and it follows logically from the
    problem statement plus preceding steps as written. An error in an earlier step
    does NOT make subsequent steps invalid by itself.
    
    Autograder Literal Checks:
    - Preceding steps are rendered as 'Step 1: <text>', 'Step 2: <text>', ... in order.
    - The current step is rendered with the label 'Current Step:'.
    - The prompt contains the phrase 'assume all preceding steps are correct' (case-insensitive).
    - The prompt contains the literal strings 'Verdict: VALID', 'Verdict: INVALID', and 'Rationale:'.
    - With an empty preceding_steps list, the prompt states '(no preceding steps)'.
    """
    if preceding_steps:
        # Rendered verbatim as "Step i: <text>" (1-indexed, in order).
        preceding = "\n".join(
            f"Step {i}: {step}" for i, step in enumerate(preceding_steps, start=1)
        )
    else:
        preceding = NO_PRECEDING_STEPS

    return JUDGE_PROMPT_TEMPLATE.format(
        question=str(question).strip(),
        preceding=preceding,
        current_step=current_step,
    )


def parse_judge_decision(judge_output: str) -> tuple[float, str]:
    r"""
    Parses the judge model's raw text completion into (score, rationale).
    
    Rules:
      1. If a line matches /verdict\s*:\s*\**\s*(VALID|INVALID)\b/i, use it.
         If several exist, use the first.
      2. Otherwise, if standalone word INVALID or phrase 'NOT VALID' appears
         (case-insensitive, word boundaries) -> -1.0.
      3. Otherwise, if standalone word VALID appears -> +1.0.
      4. Otherwise -> -1.0 (unparseable defaults to INVALID).
      
    Warning: A naive '"VALID" in text' check is wrong because "INVALID" contains "VALID".
    
    Rationale: Text after 'Rationale:' up to the first sentence end (".", "!", or "?"
    followed by whitespace or end of string, so decimals like "2.5" are not split),
    or a newline, whichever comes first, stripped. If absent, returns empty string "".
    """
    if not isinstance(judge_output, str):
        return -1.0, ""
    return _parse_score(judge_output), _parse_rationale(judge_output)


def _parse_score(text: str) -> float:
    # Rule 1: the first line containing an explicit verdict decides.
    for line in text.splitlines():
        m = _VERDICT_LINE_RE.search(line)
        if m:
            return 1.0 if m.group(1).upper() == "VALID" else -1.0
    # Rule 2: any standalone INVALID / NOT VALID -> invalid (checked BEFORE VALID).
    if _NEGATIVE_RE.search(text):
        return -1.0
    # Rule 3: standalone VALID.
    if _POSITIVE_RE.search(text):
        return 1.0
    # Rule 4: unparseable defaults to INVALID.
    return -1.0


def _parse_rationale(text: str) -> str:
    m = _RATIONALE_LABEL_RE.search(text)
    if m is None:
        return ""
    # Stop at the newline first, then at the first sentence end within that line.
    line = text[m.end():].split("\n", 1)[0]
    end = _SENTENCE_END_RE.search(line)
    if end is not None:
        line = line[:end.end()]  # keep the punctuation mark
    return line.strip()


def score_path(
    judge_model: Any = None, 
    judge_tokenizer: Any = None, 
    question: str = "", 
    candidate: dict[str, Any] | None = None, 
    judge_fn: JudgeFn | None = None
) -> dict[str, Any]:
    """
    Sequentially scores each step in candidate["steps"] using judge_fn.
    Step t is judged with preceding_steps = steps[:t].
    
    The student's code MUST call build_prm_judge_prompt to build the string passed to judge_fn,
    and MUST call parse_judge_decision on the returned text.
    
    Returns:
      step_scores, rationales, judge_calls, judge_tokens
    """
    if judge_fn is None:
        if judge_model is None or judge_tokenizer is None:
            raise ValueError("Provide either judge_fn or both judge_model and judge_tokenizer.")
        judge_fn = make_hf_judge_fn(judge_model, judge_tokenizer)

    steps = list((candidate or {}).get("steps") or [])
    step_scores: list[float] = []
    rationales: list[str] = []
    judge_tokens = 0

    # Every step is judged (no early stopping), each with ALL earlier steps as context.
    for t, current_step in enumerate(steps):
        prompt = build_prm_judge_prompt(question, steps[:t], current_step)
        out = judge_fn(prompt)
        score, rationale = parse_judge_decision(out.get("text", ""))
        step_scores.append(score)
        rationales.append(rationale)
        judge_tokens += int(out.get("num_tokens", 0))

    return {
        "step_scores": step_scores,
        "rationales": rationales,
        "judge_calls": len(steps),  # one judged step = one judge call
        "judge_tokens": judge_tokens,
    }
