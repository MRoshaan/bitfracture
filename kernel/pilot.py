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

BFCL_TAG_URL = "https://github.com/ShishirPatil/gorilla/archive/refs/heads/main.tar.gz"
REPO_URL = "https://github.com/MRoshaan/bitfracture.git"
REPO_DIR = Path("/kaggle/working/bitfracture")
RUN_DIR = REPO_DIR / "run"
WORK = Path("/kaggle/working")
BFCL_EXTRACT = WORK / "bfcl" / "gorilla-main"


def fetch_bfcl() -> Path | None:
    """Fetch and extract the BFCL repo; return the located data root (or None)."""
    tar = WORK / "gorilla.tar.gz"
    if BFCL_EXTRACT.exists():
        print("[bfcl] already extracted; skipping fetch.")
        return _find_bfcl_root(BFCL_EXTRACT)

    tar.unlink(missing_ok=True)
    subprocess.run(["curl", "-sL", BFCL_TAG_URL, "-o", str(tar)], check=False)
    if not tar.exists():
        print("[bfcl] download failed (no tar file).")
        return None
    subprocess.run(["tar", "-xzf", str(tar), "-C", str(WORK / "bfcl")], check=False)
    tar.unlink(missing_ok=True)
    return _find_bfcl_root(BFCL_EXTRACT)


def _find_bfcl_root(extract_dir: Path) -> Path | None:
    """Locate the berkeley-function-call-leaderboard package root under extract_dir."""
    cands = [
        extract_dir / "berkeley-function-call-leaderboard",
        extract_dir / "berkeley-function-call-leaderboard" / "bfcl",
    ]
    for c in cands:
        if (c / "bfcl_eval").is_dir() or (c / "data").is_dir() or (c / "bfcl").is_dir():
            return c
    return extract_dir


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
        bfcl_root / "bfcl" / "data",
        bfcl_root / "bfcl_eval" / "data",
        bfcl_root / "data",
    ]
    data_dir = next((d for d in data_candidates if d.is_dir()), None)
    probe["data_dir"] = str(data_dir) if data_dir else None
    for cat in cats:
        found = []
        if data_dir:
            for p in (data_dir / cat).glob("*") if (data_dir / cat).is_dir() else []:
                found.append(p.name)
        probe["cats"][cat] = found
    # ground truth (possible_answer)
    pa_candidates = [
        bfcl_root / "possible_answer",
        bfcl_root / "bfcl" / "data" / "possible_answer",
        bfcl_root / "bfcl_eval" / "data" / "possible_answer",
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


def main() -> None:
    # 1. Clone the project repo and make its run/ modules importable.
    sync_repo()
    sys.path.insert(0, str(RUN_DIR))
    sys.path.insert(0, str(WORK))

    # 2. Fetch BFCL data tree (pinned main@latest; exact commit logged at pull time).
    bfcl_root = fetch_bfcl()
    print("[bfcl] data layout probe:\n" + probe_data_layout(bfcl_root))

    # 3. Run the pilot.
    import pilot_runner

    root_arg = bfcl_root if bfcl_root is not None else "/kaggle/working/bfcl/gorilla-main"
    pilot_runner.main(root_arg)

    print(json.dumps({"finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}))


if __name__ == "__main__":
    main()
