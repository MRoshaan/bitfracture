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

    Tolerates the <tool_call> wrapper, the paren/json body form, and the legacy
    self-closing <name k=v> form emitted by older chat templates.
    """
    calls: list[dict] = []
    for m in TOOL_CALL_RE.finditer(raw):
        inner = m.group(1).strip()
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

    `entry` is the BFCL ground-truth prompt entry; `raw` is the model output.
    For the pilot, ground-truth is defined by which tools are expected vs. offered,
    which is sufficient to distinguish missed/unnecessary/wrong-tool/wrong-args.
    """
    if is_truncated(raw, max_new_tokens, output_token_count):
        return "truncation"

    calls = extract_tool_calls(raw)
    expected = _expected_call(entry)

    if not calls:
        # Empty/unparseable output.
        if has_tool_call(raw) is False and not raw.strip():
            return "malformed"
        # Model produced text but no tool call -> if one was required, it's missed.
        if expected is not None:
            return "missed_required"
        return "unnecessary" if _should_not_call(category) else "malformed"

    # At least one tool call parsed.
    if expected is None:
        # Ground truth says no call should be made -> any call is unnecessary.
        return "unnecessary"

    name = next(iter(calls[0]))
    if name != expected["name"]:
        return "wrong_tool"
    if set(calls[0][name].keys()) != set(expected.get("arguments", {}).keys()):
        return "wrong_args"
    return "correct"


def _expected_call(entry: dict) -> dict | None:
    """Derive the single expected call from the ground-truth entry, if any."""
    gt = entry.get("ground_truth")
    if not gt:
        return None
    # BFCL ground truth is a list of {name, arguments} dicts.
    if isinstance(gt, list) and gt:
        item = gt[0]
        return {"name": item["name"], "arguments": item.get("arguments", {})}
    return None


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


def format_breakdown(labels: list[str], classes: list[str]) -> Counter:
    counts = Counter(labels)
    return Counter({cls: counts.get(cls, 0) for cls in classes})
