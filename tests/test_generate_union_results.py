import importlib.util
from pathlib import Path

import pandas as pd


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "generate_union_results.py"
SPEC = importlib.util.spec_from_file_location("generate_union_results", SCRIPT_PATH)
generate_union_results = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(generate_union_results)


def test_generates_union_results_from_input_directory(tmp_path, monkeypatch):
    result_dir = tmp_path / "resultados" / "instancia"
    result_dir.mkdir(parents=True)
    pd.DataFrame({"file": ["a.dat"], "time": [1], "f1": [10]}).to_excel(
        result_dir / "resultado.xlsx", index=False
    )
    configured_targets_dir = tmp_path / "targets-configurados"
    configured_targets_dir.mkdir()
    pd.DataFrame(
        {
            "file": ["a.dat"],
            "time": [1],
            "f1_target": [999],
            "f2_target": [999],
            "f3_target": [999],
            "f4_target": [999],
            "f5_target": [999],
        }
    ).to_excel(configured_targets_dir / "targets.xlsx", index=False)
    pd.DataFrame({"file": ["target.dat"], "time": [1], "f1": [999]}).to_excel(
        tmp_path / "resultados" / "targets.xlsx", index=False
    )
    monkeypatch.setattr(generate_union_results, "_build_run_tag", lambda now: "test-run")
    monkeypatch.setattr(
        "src.process.PostProcessingProcess.Config.get_nested",
        lambda *keys, default=None: (
            str(configured_targets_dir)
            if keys == ("postprocessing", "output")
            else default
        ),
    )

    exit_code = generate_union_results.main(["--input-dir", str(tmp_path / "resultados")])

    union_path = tmp_path / "resultados" / "test-run-union_results.xlsx"
    assert exit_code == 0
    assert union_path.exists()
    union_df = pd.read_excel(union_path)
    assert union_df["file"].tolist() == ["a.dat"]
    assert "target.dat" not in union_df["file"].tolist()
    assert pd.isna(union_df.loc[0, "f1_target"])


def test_returns_error_when_input_directory_has_no_valid_workbooks(tmp_path, monkeypatch):
    monkeypatch.setattr(generate_union_results, "_build_run_tag", lambda now: "test-run")

    exit_code = generate_union_results.main(["--input-dir", str(tmp_path)])

    assert exit_code == 1
    assert not (tmp_path / "test-run-union_results.xlsx").exists()


def test_returns_error_when_input_directory_does_not_exist(tmp_path):
    missing_dir = Path(tmp_path) / "inexistente"

    exit_code = generate_union_results.main(["--input-dir", str(missing_dir)])

    assert exit_code == 1
