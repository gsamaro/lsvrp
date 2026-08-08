#!/usr/bin/env python3
import argparse
import json

import numpy as np

from src.helpers.ReadPrpFile import ReadPrpFile
from src.solvers.ParticleSwarmOptimization import ParticleSwarmOptimization


class QuietLogger:
    def info(self, message):
        pass

    def warning(self, message):
        print(message)

    def error(self, message):
        print(message)

    def debug(self, message):
        pass


def main():
    parser = argparse.ArgumentParser(description="Benchmark quente do PSO compilado")
    parser.add_argument(
        "--instance",
        default="data/DATA_PRP_20C/PRP10_C20_P8_V5_T12_S1.dat",
    )
    parser.add_argument("--particles", type=int, default=1000)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--target-seconds", type=float, default=60.0)
    args = parser.parse_args()

    log = QuietLogger()
    data = ReadPrpFile(args.instance, log).getDataSet()
    solver = ParticleSwarmOptimization(data, "/tmp", log)
    solver.pso_config["swarm_size"] = args.particles
    solver.pso_config["max_iterations"] = args.iterations
    solver.pso_config["return_heuristic_result_without_cplex"] = True
    solver.pso_config["use_as_mip_start"] = False
    solver.solver(numThreads=1, timeLimit=60)

    iteration_times = [
        metrics["elapsed_seconds"] for metrics in solver._population_history[1:]
    ]
    result = {
        "instance": args.instance,
        "particles": args.particles,
        "iterations": args.iterations,
        "parallel_workers": solver.parallel_workers,
        "bounds_seconds": solver.bounds_seconds,
        "jit_warmup_seconds": solver.jit_warmup_seconds,
        "initialization_seconds": solver.initialization_seconds,
        "pso_hot_elapsed_seconds": solver.pso_hot_elapsed_seconds,
        "pso_cold_elapsed_seconds": solver.pso_cold_elapsed_seconds,
        "iteration_mean_seconds": float(np.mean(iteration_times)),
        "iteration_p95_seconds": float(np.percentile(iteration_times, 95)),
        "iteration_max_seconds": float(np.max(iteration_times)),
        "best_cost": solver.global_best_cost,
        "best_first_seen_iteration": solver._best_first_seen_iteration,
        "target_seconds": args.target_seconds,
        "target_met": solver.pso_hot_elapsed_seconds <= args.target_seconds,
    }
    print(json.dumps(result, indent=2))
    if not result["target_met"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
