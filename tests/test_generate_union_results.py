import importlib.util
from pathlib import Path

import pandas as pd

from src.process.PostProcessingProcess import PostProcessingProcess


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "generate_union_results.py"
SPEC = importlib.util.spec_from_file_location("generate_union_results", SCRIPT_PATH)
generate_union_results = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(generate_union_results)


def test_generates_union_results_from_input_directory(tmp_path, monkeypatch):
    result_dir = tmp_path / "resultados" / "instancia"
    result_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "file": ["a.dat"],
            "time": [1],
            "f1": [10],
            "commit_hash": ["012345"],
        }
    ).to_excel(result_dir / "resultado.xlsx", index=False)
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
    union_df = pd.read_excel(union_path, dtype={"commit_hash": str})
    assert union_df["file"].tolist() == ["a.dat"]
    assert union_df["commit_hash"].tolist() == [
        "012345"
    ]
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


def test_union_adds_empty_commit_hash_for_legacy_workbooks(tmp_path, monkeypatch):
    pd.DataFrame({"file": ["legacy.dat"], "time": [1], "f1": [10]}).to_excel(
        tmp_path / "legacy.xlsx", index=False
    )
    monkeypatch.setattr(
        "src.process.PostProcessingProcess.Config.get_nested",
        lambda *keys, default=None: default,
    )

    processor = PostProcessingProcess(log=None, output=str(tmp_path))
    union_path = processor.union_results(run_tag="test-run")

    union_df = pd.read_excel(
        union_path,
        engine="openpyxl",
        dtype={"commit_hash": str, "config_hash": str},
    )
    configs_df = pd.read_excel(
        union_path, sheet_name="run_configs", engine="openpyxl"
    )
    assert "commit_hash" in union_df.columns
    assert pd.isna(union_df.loc[0, "commit_hash"])
    assert "config_hash" in union_df.columns
    assert pd.isna(union_df.loc[0, "config_hash"])
    assert list(configs_df.columns) == ["config_hash", "config_json"]
    assert configs_df.empty


def test_union_deduplicates_run_config_metadata(tmp_path, monkeypatch):
    configs = [
        ("a" * 64, '{"solver":{"method":"PSO"}}'),
        ("a" * 64, '{"solver":{"method":"PSO"}}'),
        ("b" * 64, '{"solver":{"method":"GUROBY"}}'),
    ]
    for index, (config_hash, config_json) in enumerate(configs):
        result_path = tmp_path / f"result-{index}.xlsx"
        with pd.ExcelWriter(result_path, engine="openpyxl") as writer:
            pd.DataFrame(
                {
                    "file": [f"instance-{index}.dat"],
                    "time": [1],
                    "config_hash": [config_hash],
                }
            ).to_excel(writer, index=False)
            pd.DataFrame(
                [{"config_hash": config_hash, "config_json": config_json}]
            ).to_excel(writer, sheet_name="run_configs", index=False)

    monkeypatch.setattr(
        "src.process.PostProcessingProcess.Config.get_nested",
        lambda *keys, default=None: default,
    )
    processor = PostProcessingProcess(log=None, output=str(tmp_path))
    union_path = processor.union_results(run_tag="test-run")

    union_df = pd.read_excel(
        union_path,
        engine="openpyxl",
        dtype={"config_hash": str},
    )
    configs_df = pd.read_excel(
        union_path, sheet_name="run_configs", engine="openpyxl"
    )
    assert set(union_df["config_hash"]) == {"a" * 64, "b" * 64}
    assert len(configs_df) == 2
    assert configs_df.set_index("config_hash").loc["a" * 64, "config_json"] == (
        '{"solver":{"method":"PSO"}}'
    )
