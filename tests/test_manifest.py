from pathlib import Path

from pka_calculator.manifest import infer_legacy_xyz_manifest


def test_legacy_import_suffixes(tmp_path: Path):
    for name in ["1.xyz", "1_H0_deprotonated.xyz", "1_H1_protonated.xyz"]:
        (tmp_path / name).write_text("1\nx\nH 0 0 0\n")
    states = infer_legacy_xyz_manifest(tmp_path)
    by_state = {s.state: s for s in states}
    assert by_state["neutral"].charge == 0
    assert by_state["deprotonated"].charge == -1
    assert by_state["protonated"].charge == 1
