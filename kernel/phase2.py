"""BitFracture Phase 2 Kaggle entrypoint (powered run).

Same pipeline as the pilot kernel, with the pre-registered Phase 2 design:
  - both models (Qwen3-1.7B + Qwen3-4B) x {fp16, nf4}
  - 30 entries per category x 5 categories = 150 per model-format
  - writes results_phase2/phase2_results.json + phase2_summary.txt

All artifacts under /kaggle/working/results_phase2/; repo clone + BFCL
extract are cleaned up in a finally block so pulled output stays lean.
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


def sync_repo() -> Path:
    """Clone the project repo so its run/ modules are importable in-kernel."""
    if (REPO_DIR / "run" / "phase2_runner.py").exists():
        print("[repo] already present; skipping clone.")
        return REPO_DIR
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(REPO_DIR)], check=False)
    return REPO_DIR


def ensure_deps() -> None:
    """Upgrade bitsandbytes so transformers 5.0.0 can load 4-bit (NF4)."""
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "-U", "bitsandbytes>=0.46.1"],
        check=False,
    )
    import importlib.metadata as md

    try:
        print(f"[deps] bitsandbytes -> {md.version('bitsandbytes')}")
    except Exception:
        print("[deps] bitsandbytes version unknown")


def fetch_bfcl() -> Path | None:
    """Fetch and extract the pinned BFCL repo; return the data root."""
    existing = _find_bfcl_root(BFCL_PARENT)
    if existing is not None:
        print(f"[bfcl] already extracted at {existing}; skipping fetch.")
        return existing
    tar = WORK / "gorilla.tar.gz"
    tar.unlink(missing_ok=True)
    subprocess.run(["curl", "-sL", BFCL_TAG_URL, "-o", str(tar)], check=False)
    if not tar.exists():
        print("[bfcl] download failed.")
        return None
    BFCL_PARENT.mkdir(parents=True, exist_ok=True)
    subprocess.run(["tar", "-xzf", str(tar), "-C", str(BFCL_PARENT)], check=False)
    tar.unlink(missing_ok=True)
    return _find_bfcl_root(BFCL_PARENT)


def _find_bfcl_root(search_root: Path) -> Path | None:
    for cand in search_root.rglob("berkeley-function-call-leaderboard"):
        if cand.is_dir():
            return cand
    return None


def main() -> None:
    sync_repo()
    sys.path.insert(0, str(RUN_DIR))
    ensure_deps()

    bfcl_root = fetch_bfcl()
    print(f"[bfcl] root: {bfcl_root}")
    if bfcl_root is None:
        raise RuntimeError("BFCL data tree could not be located — aborting phase 2.")

    import phase2_runner

    try:
        phase2_runner.main(bfcl_root)
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
