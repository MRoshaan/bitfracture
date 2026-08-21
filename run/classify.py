"""BitFracture Phase 1 — 7-class error taxonomy classifier + bootstrap CIs.

Our core contribution is a taxonomy-level audit: classify each generated tool
response into one of seven error classes (or "correct"), then compare the
distribution between quantization formats. The gate decision in analysis/GATE.md
rests on whether the shift is outside bootstrap confidence intervals.

The taxonomy (ROADMAP.md:32-41):
  1. missed_required   tool was required but never called
  2. unnecessary       called a tool when none needed
  3. wrong_tool        called a tool that isn't the right one
  4. wrong_args        correct tool, wrong/bad arguments
  5. malformed         output not parseable as a function call
  6. failed_exec       valid parse but execution state fails
  7. truncation        output cut off by max tokens

We parse the Qwen3 native tool-call format (<tool_call>...</tool_call>) with an
independent parser (not BFCL's AST parser) so the taxonomy is definitionally our
own and does not depend on BFCL model registration.
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter

TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
# Qwen3 can emit the legacy self-closing form: <get_weather city=Tokyo>
LEGACY_TAG_RE = re.compile(r"<(\w+)\s+([^>]*)>", re.DOTALL)
# Inside a tool_call body:  name({json})   or   name(k=v, ...)
CALL_BODY_RE = re.compile(r"(\w+)\s*\(\s*\{(.*?)\}\s*\)", re.DOTALL)


def extract_tool_calls(raw: str) -> list[dict]:
    """Parse Qwen3-native tool calls into a list of {name: {args}} dicts.

    Handles the JSON form  {"name": ..., "arguments": {...}}  inside <tool_call>,
    the paren/json body form  name({json}), and the legacy self-closing
    <name k=v> form emitted by older chat templates.
    """
    calls: list[dict] = []
    for m in TOOL_CALL_RE.finditer(raw):
        inner = m.group(1).strip()
        parsed = _json_form(inner)
        if parsed is not None:
            calls.extend(parsed)
            continue
        body = re.search(CALL_BODY_RE, inner)
        if body:
            name, args_text = body.group(1), body.group(2).strip()
            calls.append({name: _loose_json(args_text)})
        else:
            calls.append(_legacy_or_unparseable(inner))
    # Also catch legacy self-closing tags outside a <tool_call> wrapper.
    if not calls:
        for m in LEGACY_TAG_RE.finditer(raw):
            name, attrs = m.group(1), m.group(2).strip()
            if not attrs:
                calls.append({name: {}})
            else:
                calls.append({name: _loose_kv(attrs)})
    return calls


def _json_form(inner: str) -> list[dict] | None:
    """Parse a Qwen3 JSON tool-call body like {"name":..., "arguments":{}}.

    Returns a list of {name: {args}} dicts, or None if the text is not in this
    form (so the caller can fall back to other parsers).
    """
    text = inner.strip()
    text_arr = text if text.startswith("[") else f"[{text}]"
    try:
        obj = json.loads(text_arr)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, list):
        return None
    calls = []
    for item in obj:
        if not isinstance(item, dict) or "name" not in item:
            return None
        args = item.get("arguments", {})
        calls.append({item["name"]: args if isinstance(args, dict) else {}})
    return calls


def _legacy_or_unparseable(inner: str) -> dict:
    inner = inner.strip()
    if inner.startswith("<") and inner.endswith(">"):
        m = re.match(r"<(\w+)\s+([^>]*)>", inner, re.DOTALL)
        if m:
            name, attrs = m.group(1), m.group(2).strip()
            if attrs:
                return {name: _loose_kv(attrs)}
            return {name: {}}
    return _unparseable(inner)


def _loose_kv(attrs: str) -> dict:
    """Parse `k=v, k2=v2` attr text into {k: v} with unquoted best-effort values."""
    out: dict[str, object] = {}
    for part in attrs.split(","):
        part = part.strip()
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        out[k.strip()] = v.strip().strip("'\"")
    return out


def _unparseable(inner: str) -> dict:
    # Represent an unparseable block minimally so downstream code treats it as malformed.
    return {"__malformed__": inner}


def _loose_json(args_text: str) -> dict:
    """Best-effort parse of a JSON-ish argument body. Returns {} on failure (not fatal)."""
    s = args_text.strip()
    if not s:
        return {}
    if not s.startswith("{"):
        s = "{" + s + "}"
    try:
        value = json.loads(s)
        return value if isinstance(value, dict) else {}
    except json.JSONDecodeError:
        # Not clean JSON; best-effort attr-style parse is applied by callers where relevant.
        return {}


def is_truncated(raw: str, max_new_tokens: int, output_token_count: int) -> bool:
    """Heuristic: emitted at/over the generation budget *and* not self-terminated."""
    if output_token_count <= 0:
        return False
    return output_token_count >= max_new_tokens and not raw.rstrip().endswith((">", "}"))


def has_tool_call(raw: str) -> bool:
    return bool(TOOL_CALL_RE.search(raw))


def classify(
    entry: dict,
    raw: str,
    category: str,
    max_new_tokens: int,
    output_token_count: int = 0,
) -> str:
    """Return one of the taxonomy class names, or 'correct'.

    `entry` is the BFCL prompt entry (with ground_truth merged in); `raw` is the
    model output. For the pilot, ground-truth correctness is decided at the
    tool-name + argument-key level (which tool, and which of its arguments the
    model supplied), sufficient to separate correct / missed / wrong_tool /
    wrong_args / unnecessary.
    """
    if is_truncated(raw, max_new_tokens, output_token_count):
        return "truncation"

    calls = extract_tool_calls(raw)
    expected = _expected_call(entry)

    if not calls:
        if not raw.strip():
            return "malformed"
        if expected is not None:
            return "missed_required"
        return "unnecessary" if _should_not_call(category) else "malformed"

    if expected is None:
        return "unnecessary"

    name = next(iter(calls[0]))
    if name != expected["name"]:
        return "wrong_tool"

    model_args = set(calls[0][name].keys())
    if model_args != expected["argument_keys"]:
        return "wrong_args"
    return "correct"


def _expected_call(entry: dict) -> dict | None:
    """Derive the single expected call from the BFCL ground-truth entry, if any.

    BFCL ground truth is a list like [{"tool_name": {"arg": [allowed values]}}].
    We track the expected tool name and the set of argument keys it should carry.
    """
    gt = entry.get("ground_truth")
    if not gt or not isinstance(gt, list) or not gt:
        return None
    item = gt[0]
    if not isinstance(item, dict):
        return None
    name = next(iter(item))
    arg_spec = item[name]
    keys = set(arg_spec.keys()) if isinstance(arg_spec, dict) else set()
    # Drop placeholder keys used to mark optional params; keep real ones.
    keys = {k for k in keys if k}
    return {"name": name, "argument_keys": keys}


def _should_not_call(category: str) -> bool:
    # relevance/irrelevance categories expect the model to avoid calls.
    return "irrelevance" in category


def bootstrap_ci(
    labels: list[str],
    classes: list[str],
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, dict[str, float]]:
    """Per-class proportion with bootstrap percentile CI over n_boot resamples."""
    rng = random.Random(seed)
    n = len(labels)
    result: dict[str, dict[str, float]] = {}
    for cls in classes:
        means = []
        for _ in range(n_boot):
            sample = [rng.choice(labels) for _ in range(n)]
            means.append(sample.count(cls) / n)
        sorted_means = sorted(means)
        lo = sorted_means[int(n_boot * alpha / 2)]
        hi = sorted_means[int(n_boot * (1 - alpha / 2))]
        result[cls] = {
            "mean": sorted_means[n_boot // 2],
            "ci_lo": lo,
            "ci_hi": hi,
            "count": labels.count(cls),
        }
    return result


def bootstrap_diff_ci(
    labels_a: list[str],
    labels_b: list[str],
    classes: list[str],
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, dict[str, float]]:
    """Two-sample bootstrap CI on the proportion difference (a - b) per class.

    Resamples each group independently so the CI reflects uncertainty in both
    formats. Positive delta means the class is MORE common in group a.
    """
    rng = random.Random(seed)
    na, nb = len(labels_a), len(labels_b)
    result: dict[str, dict[str, float]] = {}
    for cls in classes:
        diffs = []
        for _ in range(n_boot):
            sa = sum(rng.choice(labels_a) == cls for _ in range(na)) / na
            sb = sum(rng.choice(labels_b) == cls for _ in range(nb)) / nb
            diffs.append(sa - sb)
        sorted_diffs = sorted(diffs)
        result[cls] = {
            "mean": labels_a.count(cls) / na - labels_b.count(cls) / nb,
            "ci_lo": sorted_diffs[int(n_boot * alpha / 2)],
            "ci_hi": sorted_diffs[int(n_boot * (1 - alpha / 2))],
        }
    return result


def format_breakdown(labels: list[str], classes: list[str]) -> Counter:
    counts = Counter(labels)
    return Counter({cls: counts.get(cls, 0) for cls in classes})
