"""Repo integrity checks that run without a GPU (CI-safe)."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_core_docs_exist() -> None:
    for name in ("ROADMAP.md", "README.md", ".gitignore"):
        assert (ROOT / name).exists(), f"missing {name}"


def test_results_ignored() -> None:
    ignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "results/" in ignore, "results/ must be gitignored (no raw outputs in git)"


def test_pins_requirements_present() -> None:
    req = ROOT / "pins" / "requirements.txt"
    assert req.exists(), "pins/requirements.txt is missing"
    lines = [
        ln.strip()
        for ln in req.read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]
    assert lines, "pins/requirements.txt must list at least one dependency"


def test_reference_seed_present() -> None:
    seed = ROOT / "reference" / "neurips2026_slm_agentic_ideas.md"
    assert seed.exists(), "source ideas doc missing from reference/"


def test_pilot_metadata_points_to_pilot_script() -> None:
    import json

    meta = json.loads((ROOT / "kernel" / "pilot-metadata.json").read_text(encoding="utf-8"))
    code_file = meta["code_file"]
    assert (ROOT / "kernel" / code_file).exists(), (
        f"kernel/{code_file} referenced by pilot-metadata.json but missing"
    )
    assert meta["enable_gpu"] is True
