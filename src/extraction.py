"""
src/extraction.py - Step and answer extraction utilities.
"""
import re


def parse_steps(completion_text: str) -> list[str]:
    """
    Parses completion text and returns a list of individual step strings.
    Must tolerate markdown decoration around labels (e.g. '**Step 1:**', '### Step 2:')
    and strip leftover '*' or '#' from captured text.
    """
    if not completion_text or not isinstance(completion_text, str):
        return []

    # Pattern matching Step markers with optional markdown decoration (#, *, `, :, .)
    pattern = r"(?:^|\n)\s*(?:[#*`\s]*Step\s*\d+\s*[:.]?[#*`\s]*)(.*?)(?=(?:\n\s*[#*`\s]*Step\s*\d+\s*[:.]?[#*`\s]*)|(?:\n\s*[#*`\s]*Final Answer)|$)"
    matches = re.findall(pattern, completion_text, flags=re.DOTALL | re.IGNORECASE)
    cleaned = []
    for m in matches:
        text = m.strip().strip("*#`").strip()
        if text:
            cleaned.append(text)
    return cleaned


def extract_answer(completion_text: str) -> int | None:
    """
    Extracts the final numeric integer from the completion.

    Required Behaviors:
      - 'Final Answer: 18' -> 18
      - 'Final Answer: $1,234' -> 1234
      - 'Final Answer: 18.00' -> 18
      - 'Final Answer: 18.5' -> None (non-integer)
      - 'Final Answer: -5' -> -5
      - 'Final Answer: 18 dollars.' -> 18
      - '**Final Answer:** 18' -> 18
      - 'final answer 18' (case-insensitive, colon optional) -> 18
      - Multiple 'Final Answer' lines -> use the LAST one
      - No 'Final Answer' present -> None
      - Empty string or None-like garbage -> None, and never raise
    """
    try:
        return _extract_answer_impl(completion_text)
    except Exception:
        # Contract: never raise on malformed input.
        return None


# "Final Answer" delimiter: case-insensitive, any whitespace between the words.
_DELIMITER_RE = re.compile(r"final\s*answer\b", re.IGNORECASE)

# Decoration between the delimiter and the value: colon, markdown bold/headers,
# backticks, '=', the word "is", whitespace. Newlines are allowed here so that
# "Final Answer:\n18" still works.
_SEPARATOR_RE = re.compile(r"(?:[\s*_`#:=]|\bis\b)*", re.IGNORECASE)

# The value itself, anchored at the start of the delimiter's line:
#   - optional wrappers the model sometimes adds: $, \$, \boxed{, \( , \[
#   - optional sign (ASCII '-' or Unicode minus), before or after a currency sign
#   - integer part with valid thousands grouping (1,234,567) or plain digits
#   - optional decimal part (only accepted if it is all zeros)
_VALUE_RE = re.compile(
    r"(?:\\boxed\s*\{|\\[(\[]|\\?\$|\s)*"
    r"(?P<sign>[-−])?\s*(?:\\?\$\s*)?"
    r"(?P<int>\d{1,3}(?:,\d{3})+(?!\d)|\d+)"
    r"(?:\.(?P<frac>\d+))?"
    r"(?P<after>.*)"
)

# If the number is immediately followed by an operator and another number
# (e.g. "Final Answer: 20 - 2 = 18" or "1/2"), the line is an expression, not an answer.
# Also rejects malformed digit grouping such as "12,34" (would otherwise parse as 12).
_EXPRESSION_TAIL_RE = re.compile(r"^\s*[-+*/×÷^=]\s*[\d$(]|^,\d")


def _extract_answer_impl(completion_text) -> int | None:
    if not isinstance(completion_text, str) or not completion_text.strip():
        return None

    # 1. Locate the LAST "Final Answer" delimiter; earlier ones are scratchpad attempts.
    delimiters = list(_DELIMITER_RE.finditer(completion_text))
    if not delimiters:
        return None
    rest = completion_text[delimiters[-1].end():]

    # 2. Skip decoration (":", "**", "is", ...) and keep only the value's line.
    rest = rest[_SEPARATOR_RE.match(rest).end():]
    value_line = rest.split("\n", 1)[0]

    # 3. Parse a number at the start of that line. Anything else -> None (no fallback).
    m = _VALUE_RE.match(value_line)
    if m is None:
        return None
    if _EXPRESSION_TAIL_RE.match(m.group("after")):
        return None

    frac = m.group("frac")
    if frac is not None and int(frac) != 0:
        return None  # 18.5 is not an integer answer; 18.00 is.

    value = int(m.group("int").replace(",", ""))
    return -value if m.group("sign") else value
