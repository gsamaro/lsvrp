"""Run the focused PSO/MIP ablation experiment for one PRP instance."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


INSTANCE = "data/DATA_PRP_20C/PRP13_C20_P8_V5_T12_S4.dat"
WEIGHT = [0.2] * 5
ALPHA = 0.99
TIME_LIMIT_SECONDS = int(os.environ.get("LSVRP_ABLATION_WALL_SECONDS", "180"))
MIP_TIME_LIMIT_SECONDS = int(os.environ.get("LSVRP_ABLATION_MIP_SECONDS", "120"))
PSO_MAX_ITERATIONS = int(os.environ.get("LSVRP_ABLATION_PSO_ITERATIONS", "100"))
INTERNAL_THREADS = os.environ.get("LSVRP_ABLATION_INTERNAL_THREADS")

SCENARIOS = [
    {"id": "S1", "strengthened_bounds": False, "coelho": False, "cuts": False},
    {"id": "S2", "strengthened_bounds": True, "coelho": False, "cuts": False},
    {"id": "S3", "strengthened_bounds": False, "coelho": True, "cuts": False},
    {"id": "S4", "strengthened_bounds": True, "coelho": True, "cuts": False},
    {"id": "S5", "strengthened_bounds": False, "coelho": False, "cuts": True},
    {"id": "S6", "strengthened_bounds": True, "coelho": False, "cuts": True},
    {"id": "S7", "strengthened_bounds": False, "coelho": True, "cuts": True},
    {"id": "S8", "strengthened_bounds": True, "coelho": True, "cuts": True},
]


def _configure(scenario):
    from config import Config

    Config.load()
    Config._data["solver"]["multiobjective"] = True
    Config._data["solver"]["coelho_inequalities"] = scenario["coelho"]
    Config._data["solver"]["rounded_capacity_inequalities"]["enabled"] = scenario[
        "cuts"
    ]
    Config._data["solver"]["pso"]["parallel_workers"] = 1
    Config._data["solver"]["pso"]["max_iterations"] = PSO_MAX_ITERATIONS
    Config._data["solver"]["pso"]["use_as_mip_start"] = True
    Config._data["solver"]["pso"]["return_heuristic_result_without_cplex"] = False
    Config._data["solver"]["threadsLimit"] = 1
    if os.environ.get("LSVRP_ABLATION_TELEMETRY", "0") == "1":
        Config._data["solver"]["telemetry"]["enabled"] = True


def _worker(scenario, output_dir):
    from src.helpers.ReadPrpFile import ReadPrpFile
    from src.helpers.TargetsLoader import (
        load_targets_by_file,
        normalize_instance_file_key,
    )
    from src.log.Logger import Logger
    from src.solvers.ParticleSwarmOptimization import ParticleSwarmOptimization

    _configure(scenario)
    output_dir.mkdir(parents=True, exist_ok=True)
    log = Logger(
        log_dir=str(output_dir / "logs"),
        log_file="Worker_0.log",
        worker_id=0,
        task=scenario["id"],
    )
    started = time.perf_counter()
    solver = None
    result = {
        "scenario": scenario,
        "instance": INSTANCE,
        "weight": WEIGHT,
        "alpha": ALPHA,
        "time_limit_seconds": TIME_LIMIT_SECONDS,
        "mip_time_limit_seconds": MIP_TIME_LIMIT_SECONDS,
        "pso_max_iterations": PSO_MAX_ITERATIONS,
        "status": "started",
    }
    try:
        data = ReadPrpFile(file_path=INSTANCE, log=log).getDataSet()
        data["weight"] = WEIGHT
        data["alpha"] = ALPHA
        data["strengthened_bounds"] = scenario["strengthened_bounds"]
        targets = load_targets_by_file("out/target/targets.xlsx", log=log)
        key = normalize_instance_file_key(data["file"])
        data["targets"] = targets.get(key, [])
        solver = ParticleSwarmOptimization(
            map=data, dir=str(output_dir) + os.sep, log=log
        )
        solver.solver(
            timeLimit=MIP_TIME_LIMIT_SECONDS,
            numThreads=int(INTERNAL_THREADS) if INTERNAL_THREADS else None,
        )
        telemetry = solver.get_telemetry()
        relaxed = solver.relaxed_total_bounds or {}
        result.update(
            {
                "status": telemetry.get("status", "completed"),
                "telemetry": telemetry,
                "relaxed_bounds": {
                    key: relaxed.get(key)
                    for key in ("lower", "upper", "lower_gap", "upper_gap")
                    if key in relaxed
                },
                "elapsed_wall_seconds": time.perf_counter() - started,
            }
        )
    except Exception as error:
        result.update(
            {
                "status": "error",
                "error": repr(error),
                "elapsed_wall_seconds": time.perf_counter() - started,
            }
        )
        log.error(f"Falha no cenário {scenario['id']}: {error}")
    finally:
        if solver is not None:
            try:
                solver.terminate()
            except Exception as error:
                result["terminate_error"] = repr(error)
        (output_dir / "summary.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )


def _run_all(output_root):
    output_root.mkdir(parents=True, exist_ok=True)
    processes = {}
    started = time.monotonic()
    selected = os.environ.get("LSVRP_ABLATION_SCENARIOS")
    scenario_ids = {item.strip() for item in selected.split(",")} if selected else None
    scenarios = [
        scenario for scenario in SCENARIOS
        if scenario_ids is None or scenario["id"] in scenario_ids
    ]
    for scenario in scenarios:
        scenario_dir = output_root / scenario["id"]
        command = [
            sys.executable,
            __file__,
            "--worker",
            json.dumps(scenario),
            str(scenario_dir),
        ]
        processes[scenario["id"]] = {
            "scenario": scenario,
            "dir": scenario_dir,
            "process": subprocess.Popen(command),
        }

    for item in processes.values():
        process = item["process"]
        remaining = max(1, TIME_LIMIT_SECONDS - (time.monotonic() - started))
        try:
            process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            summary = {
                "scenario": item["scenario"],
                "instance": INSTANCE,
                "weight": WEIGHT,
                "alpha": ALPHA,
                "time_limit_seconds": TIME_LIMIT_SECONDS,
                "mip_time_limit_seconds": MIP_TIME_LIMIT_SECONDS,
                "pso_max_iterations": PSO_MAX_ITERATIONS,
                "status": "wall_timeout",
                "elapsed_wall_seconds": TIME_LIMIT_SECONDS,
            }
            item["dir"].mkdir(parents=True, exist_ok=True)
            (item["dir"] / "summary.json").write_text(
                json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", nargs=2, metavar=("SCENARIO", "OUTPUT_DIR"))
    parser.add_argument("--output-root", default="out/ablation_prp13_2026-09-19")
    args = parser.parse_args()
    if args.worker:
        _worker(json.loads(args.worker[0]), Path(args.worker[1]))
    else:
        _run_all(Path(args.output_root))


if __name__ == "__main__":
    main()
