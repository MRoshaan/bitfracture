"""BitFracture Phase 1 Kaggle entrypoint (pilot).

Steps:
  1. Make the project's run/ modules importable from /kaggle/working.
  2. Fetch the pinned BFCL repo so the pilot can read real prompt/ground-truth
     datafiles and locate the correct data root.
  3. Probe the on-disk BFCL layout and log it (single biggest de-risk).
  4. Run the pilot orchestration (fp16 vs nf4 on Qwen3-1.7B).

All artifacts are written under /kaggle/working/results_pilot/ and pulled back
to results/phase1/ locally (gitignored).
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

BFCL_COMMIT = "6ea57973c7a6097fd7c5915698c54c17c5b1b6c8"  # pinned (bfcl-eval 2026.3.23)
BFCL_TAG_URL = f"https://github.com/ShishirPatil/gorilla/archive/{BFCL_COMMIT}.tar.gz"
REPO_URL = "https://github.com/MRoshaan/bitfracture.git"
REPO_DIR = Path("/kaggle/working/bitfracture")
RUN_DIR = REPO_DIR / "run"
WORK = Path("/kaggle/working")
BFCL_PARENT = WORK / "bfcl"


def fetch_bfcl() -> Path | None:
    """Fetch and extract the pinned BFCL repo; return the located data root (or None).

    The commit tarball extracts to a versioned top-level dir (gorilla-<commit>/),
    so we search for the berkeley-function-call-leaderboard root rather than
    assuming a fixed subdir name.
    """
    existing = _find_bfcl_root(BFCL_PARENT)
    if existing is not None:
        print(f"[bfcl] already extracted at {existing}; skipping fetch.")
        return existing

    tar = WORK / "gorilla.tar.gz"
    tar.unlink(missing_ok=True)
    subprocess.run(["curl", "-sL", BFCL_TAG_URL, "-o", str(tar)], check=False)
    if not tar.exists():
        print("[bfcl] download failed (no tar file).")
        return None
    BFCL_PARENT.mkdir(parents=True, exist_ok=True)
    subprocess.run(["tar", "-xzf", str(tar), "-C", str(BFCL_PARENT)], check=False)
    tar.unlink(missing_ok=True)
    return _find_bfcl_root(BFCL_PARENT)


def _find_bfcl_root(search_root: Path) -> Path | None:
    """Recursively locate the berkeley-function-call-leaderboard package root."""
    for cand in search_root.rglob("berkeley-function-call-leaderboard"):
        if cand.is_dir():
            return cand
    return None


def probe_data_layout(bfcl_root: Path | None) -> str:
    """Return a compact dump of what single-turn data/ground-truth files exist."""
    probe: dict = {"bfcl_root": str(bfcl_root) if bfcl_root else None, "cats": {}}
    if bfcl_root is None:
        return json.dumps(probe)
    cats = [
        "simple_python",
        "parallel",
        "multiple",
        "live_simple",
        "live_multiple",
        "relevance",
        "irrelevance",
    ]
    data_candidates = [
        bfcl_root / "bfcl_eval" / "data",
        bfcl_root / "data",
    ]
    data_dir = next((d for d in data_candidates if d.is_dir()), None)
    probe["data_dir"] = str(data_dir) if data_dir else None
    for cat in cats:
        if data_dir:
            f = data_dir / f"BFCL_v4_{cat}.json"
            probe["cats"][cat] = [f.name] if f.exists() else []
    # ground truth (possible_answer)
    pa_candidates = [
        bfcl_root / "bfcl_eval" / "data" / "possible_answer",
        bfcl_root / "data" / "possible_answer",
    ]
    pa_dir = next((d for d in pa_candidates if d.is_dir()), None)
    probe["possible_answer_dir"] = str(pa_dir) if pa_dir else None
    if pa_dir:
        probe["pa_files"] = sorted(p.name for p in pa_dir.glob("*.json"))[:40]
    return json.dumps(probe, indent=2)


def sync_repo() -> Path:
    """Clone the project repo so its run/ modules are importable in-kernel."""
    if (REPO_DIR / "run" / "pilot_runner.py").exists():
        print("[repo] already present; skipping clone.")
        return REPO_DIR
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(REPO_DIR)], check=False)
    return REPO_DIR


def ensure_deps() -> None:
    """Upgrade bitsandbytes so transformers 5.0.0 can load 4-bit (NF4).

    The Kaggle base image ships an old bnb (<0.46.1) that transformers rejects
    for 4-bit quantization; the earlier env-check upgraded it in-kernel, so we
    must do the same here.
    """
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "-U", "bitsandbytes>=0.46.1"],
        check=False,
    )
    import importlib.metadata as md

    try:
        print(f"[deps] bitsandbytes -> {md.version('bitsandbytes')}")
    except Exception:
        print("[deps] bitsandbytes version unknown")


def main() -> None:
    # 1. Clone the project repo and make its run/ modules importable.
    sync_repo()
    sys.path.insert(0, str(RUN_DIR))
    sys.path.insert(0, str(WORK))

    # 1b. Ensure NF4-capable bitsandbytes.
    ensure_deps()

    # 2. Fetch pinned BFCL data tree (commit pinned above; exact SHA logged at pull).
    bfcl_root = fetch_bfcl()
    print("[bfcl] data layout probe:\n" + probe_data_layout(bfcl_root))
    if bfcl_root is None:
        raise RuntimeError("BFCL data tree could not be located — aborting pilot.")

    # 3. Run the pilot.
    import pilot_runner

    try:
        pilot_runner.main(bfcl_root)
    finally:
        _cleanup_artifacts()

    print(json.dumps({"finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}))


def _cleanup_artifacts() -> None:
    """Remove cloned repo + BFCL extract so pulled kernel output stays lean."""
    import shutil

    for p in (REPO_DIR, BFCL_PARENT):
        shutil.rmtree(p, ignore_errors=True)
    (WORK / "gorilla.tar.gz").unlink(missing_ok=True)
    print("[cleanup] removed repo clone and BFCL extract.")


if __name__ == "__main__":
    main()
