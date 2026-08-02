import time
import math

import numpy as np

from config import Config
from src.log.Logger import Logger
from src.solvers.FeasibleParticleHeuristic import FeasibleParticleHeuristic
from src.solvers.LotSizingRelaxation import LotSizingRelaxation
from src.solvers.MultProductProdctionRoutingProblem import (
    MultProductProdctionRoutingProblem as MPPRP,
)
from src.solvers._solver_common import (
    ProblemData,
    build_results_from_variables,
    clone_solution,
    empty_solution,
    evaluate_solution_cost,
)


class ParticleSwarmOptimization:
    def __init__(self, map, dir, log: Logger):
        self.data = map
        self.dir = dir
        self.log = log
        self.problem = ProblemData.from_map(map)
        self.pso_config = self._load_pso_config()
        self.relaxed_bounds = None
        self.relaxed_base = None
        self.relaxed_total_bounds = None
        bounds_config = self.pso_config["lot_sizing_bounds"]
        if bounds_config["enabled"]:
            self.log.info(">> Calculando bounds relaxados de dimensionamento de lotes.")
            time_limit = bounds_config["time_limit"]
            if time_limit is None:
                time_limit = Config.get_nested("solver", "timeLimit", default=None)
            relaxation = LotSizingRelaxation(
                map=map,
                log=log,
                time_limit=time_limit,
            )
            relaxed = relaxation.solve_bounds(
                include_base=bounds_config["use_base_solution"]
            )
            self.relaxed_total_bounds = relaxed
            self.relaxed_bounds = {
                "lower": np.asarray(relaxed["lower_solution"]["X"], dtype=float),
                "upper": np.asarray(relaxed["upper_solution"]["X"], dtype=float),
                "q_lower": np.asarray(relaxed["lower_solution"]["Q"], dtype=float),
                "q_upper": np.asarray(relaxed["upper_solution"]["Q"], dtype=float),
            }
            self.relaxed_base = relaxed.get("base") or relaxed["lower_solution"]
            self.log.info(
                f">> Bounds agregados do PL: lower={relaxed['lower']} "
                f"upper={relaxed['upper']}"
            )
        self.heuristic = FeasibleParticleHeuristic(
            map=map,
            dir=dir,
            log=log,
            bounds=self.relaxed_bounds,
            relaxed_base=self.relaxed_base,
        )
        self.rng = np.random.default_rng(seed=self.pso_config["seed"])
        self.positions = None
        self.velocities = None
        self.solutions = []
        self.personal_best_positions = None
        self.personal_best_costs = None
        self.personal_best_solutions = []
        self.global_best_position = None
        self.global_best_solution = None
        self.global_best_cost = float("inf")
        self.solverGurobi = None
        self.time = 0
        self.solCount = 0
        self._population_history = []
        self._invalid_before_repair = []
        self._resample_attempts = 0
        self._resample_infeasible = 0

    def _load_pso_config(self):
        configured = Config.get_nested("solver", "pso", default={}) or {}
        defaults = {
            "swarm_size": 20,
            "max_iterations": 50,
            "inertia": 0.7,
            "cognitive": 1.5,
            "social": 1.5,
            "seed": 123,
            "repair_on_update": True,
            "lot_sizing_bounds": {
                "enabled": True,
                "use_base_solution": False,
                "time_limit": None,
            },
            "use_as_mip_start": True,
            "return_heuristic_result_without_cplex": False,
        }
        merged = {**defaults, **configured}
        merged["lot_sizing_bounds"] = {
            **defaults["lot_sizing_bounds"],
            **(configured.get("lot_sizing_bounds") or {}),
        }
        return merged

    def _random_positions(self):
        return self.rng.normal(
            loc=0.0,
            scale=1.0,
            size=(int(self.pso_config["swarm_size"]), self.heuristic.particle_dim),
        )

    def _initialize_swarm(self):
        self.positions = self._random_positions()
        self.velocities = np.zeros_like(self.positions)
        self._seed_relaxed_base_position()
        self.solutions = self.heuristic.build_population(self.positions)
        self._repair_invalid_particles()
        self._initialize_bests()
        diversity = self._record_population_metrics(iteration=0)
        self.log.info(
            "PSO initialization "
            f"feasible={diversity['feasible']}/{len(self.solutions)} "
            f"x_profiles={diversity['x_profiles']} "
            f"q_profiles={diversity['q_profiles']}"
        )

    def _seed_relaxed_base_position(self):
        if self.relaxed_base is None or self.relaxed_bounds is None:
            return
        lower = self.relaxed_bounds["lower"]
        upper = self.relaxed_bounds["upper"]
        base_x = self.relaxed_base["X"]
        for p in range(self.problem.p):
            for t in range(self.problem.t):
                span = upper[p, t] - lower[p, t]
                if span <= 1e-9:
                    gene = 0.0
                else:
                    ratio = np.clip((base_x[p, t] - lower[p, t]) / span, 1e-6, 1 - 1e-6)
                    gene = math.log(float(ratio) / float(1 - ratio))
                self.positions[0, p * self.problem.t + t] = gene

        lower_q = self.relaxed_bounds["q_lower"]
        upper_q = self.relaxed_bounds["q_upper"]
        base_q = self.relaxed_base["Q"]
        for p in range(self.problem.p):
            for v in range(self.problem.v):
                for i in range(self.problem.i):
                    for t in range(self.problem.t):
                        span = upper_q[p, v, i, t] - lower_q[p, v, i, t]
                        if span <= 1e-9:
                            gene = 0.0
                        else:
                            ratio = np.clip(
                                (base_q[p, v, i, t] - lower_q[p, v, i, t]) / span,
                                1e-6,
                                1 - 1e-6,
                            )
                            gene = math.log(float(ratio) / float(1 - ratio))
                        index = self.heuristic.dim_x + np.ravel_multi_index(
                            (p, v, i, t),
                            (self.problem.p, self.problem.v, self.problem.i, self.problem.t),
                        )
                        self.positions[0, index] = gene

    def _repair_invalid_particles(self):
        self._invalid_before_repair.append(
            sum(not solution["feasible"] for solution in self.solutions)
        )
        self._aux_solutions = {}
        self._aux_index = []
        for index, solution in enumerate(self.solutions):
            if solution["feasible"]:
                continue
            replacement = self._resample_solution()
            self._aux_index.append(index)
            self.positions[index] = replacement["position"]
            self._aux_solutions.update({index: replacement['solution']})
        for index in self._aux_index:
            self.solutions[index] = self._aux_solutions[index]
            # self.solutions[index] = replacement["solution"]

    def _resample_solution(self):
        while True:
            self._resample_attempts += 1
            candidate_position = self.rng.normal(
                loc=0.0,
                scale=1.0,
                size=(self.heuristic.particle_dim,),
            )
            candidate_solution = self.heuristic.build_population(
                np.array([candidate_position])
            )[0]
            if candidate_solution["feasible"]:
                return {"position": candidate_position, "solution": candidate_solution}
            self._resample_infeasible += 1

    def _initialize_bests(self):
        self.personal_best_positions = np.array(self.positions, copy=True)
        self.personal_best_costs = np.array(
            [
                solution["cost"] if solution["feasible"] else float("inf")
                for solution in self.solutions
            ],
            dtype=float,
        )
        self.personal_best_solutions = [clone_solution(solution) for solution in self.solutions]

        best_index = int(np.argmin(self.personal_best_costs))
        self.global_best_position = np.array(self.personal_best_positions[best_index], copy=True)
        self.global_best_solution = clone_solution(self.personal_best_solutions[best_index])
        self.global_best_cost = float(self.personal_best_costs[best_index])

    def _update_positions(self):
        r1 = self.rng.random(size=self.positions.shape)
        r2 = self.rng.random(size=self.positions.shape)
        self.velocities = (
            self.pso_config["inertia"] * self.velocities
            + self.pso_config["cognitive"] * r1 * (self.personal_best_positions - self.positions)
            + self.pso_config["social"] * r2 * (self.global_best_position - self.positions)
        )
        self.positions = self.positions + self.velocities

    def _evaluate_swarm(self):
        self.solutions = self.heuristic.repair_population(
            self.positions,
            previous_solutions=self.solutions,
        )
        self._repair_invalid_particles()

        for index, solution in enumerate(self.solutions):
            cost = solution["cost"] if solution["feasible"] else float("inf")
            if cost < self.personal_best_costs[index]:
                self.personal_best_costs[index] = cost
                self.personal_best_positions[index] = np.array(self.positions[index], copy=True)
                self.personal_best_solutions[index] = clone_solution(solution)

        best_index = int(np.argmin(self.personal_best_costs))
        if float(self.personal_best_costs[best_index]) < self.global_best_cost:
            self.global_best_cost = float(self.personal_best_costs[best_index])
            self.global_best_position = np.array(self.personal_best_positions[best_index], copy=True)
            self.global_best_solution = clone_solution(self.personal_best_solutions[best_index])

    def _population_diversity(self):
        feasible_solutions = [
            solution for solution in self.solutions if solution["feasible"]
        ]
        x_profiles = {
            tuple(np.asarray(solution["X"], dtype=int).ravel())
            for solution in feasible_solutions
        }
        q_profiles = {
            tuple(np.asarray(solution["Q"], dtype=int).sum(axis=1).ravel())
            for solution in feasible_solutions
        }
        return {
            "feasible": len(feasible_solutions),
            "x_profiles": len(x_profiles),
            "q_profiles": len(q_profiles),
        }

    def _record_population_metrics(self, iteration):
        metrics = {
            "iteration": iteration,
            "population_size": len(self.solutions),
            "best_cost": self.global_best_cost,
            **self._population_diversity(),
        }
        self._population_history.append(metrics)
        return metrics

    def _log_final_report(self):
        if not self._population_history:
            return

        initial = self._population_history[0]
        final = self._population_history[-1]
        best_cost = min(metrics["best_cost"] for metrics in self._population_history)
        first_best_iteration = next(
            metrics["iteration"]
            for metrics in self._population_history
            if metrics["best_cost"] == best_cost
        )
        total_population_candidates = sum(
            metrics["population_size"] for metrics in self._population_history
        )
        invalid_before_repair = sum(self._invalid_before_repair)
        minimum_x_profiles = min(
            metrics["x_profiles"] for metrics in self._population_history
        )
        minimum_q_profiles = min(
            metrics["q_profiles"] for metrics in self._population_history
        )
        self.log.info(
            "PSO final "
            f"best_cost={best_cost} "
            f"best_first_seen_iteration={first_best_iteration} "
            f"final_feasible={final['feasible']}/{final['population_size']}"
        )
        self.log.info(
            "PSO final diagnostics "
            f"pre_repair_infeasible={invalid_before_repair}/{total_population_candidates} "
            f"resample_infeasible={self._resample_infeasible}/{self._resample_attempts} "
            f"x_profiles={initial['x_profiles']}->{final['x_profiles']}(min={minimum_x_profiles}) "
            f"q_profiles={initial['q_profiles']}->{final['q_profiles']}(min={minimum_q_profiles})"
        )

    def _maybe_run_exact_solver(self, numThreads=None, timeLimit=None):
        standalone = bool(self.pso_config["return_heuristic_result_without_cplex"])
        use_as_mip_start = bool(self.pso_config["use_as_mip_start"])

        if standalone or not use_as_mip_start:
            return

        if self.global_best_solution is None:
            start = {"start": False}
        else:
            start = {
                "start": True,
                "variables": {
                    "X": self.global_best_solution["X"],
                    "Y": self.global_best_solution["Y"],
                    "I": self.global_best_solution["I"],
                    "Q": self.global_best_solution["Q"],
                    "R": self.global_best_solution["R"],
                    "Z": self.global_best_solution["Z"],
                },
            }

        self.solverGurobi = MPPRP(self.data, self.dir, self.log, start)
        self.solverGurobi.solver(timeLimit=timeLimit, numThreads=numThreads)

    def solver(self, numThreads=None, timeLimit=None):
        started_at = time.time()
        self._initialize_swarm()

        for iteration in range(int(self.pso_config["max_iterations"])):
            iteration_started_at = time.time()
            self._update_positions()
            self._evaluate_swarm()
            diversity = self._record_population_metrics(iteration=iteration + 1)
            self.log.info(
                f"PSO iteration={iteration + 1} "
                f"best_cost={self.global_best_cost} "
                f"feasible={diversity['feasible']}/{len(self.solutions)} "
                f"x_profiles={diversity['x_profiles']} "
                f"q_profiles={diversity['q_profiles']} "
                f"elapsed={time.time() - iteration_started_at:.2f}s"
            )

        self.time = time.time() - started_at
        self.solCount = 1 if self.global_best_solution is not None else 0
        self._log_final_report()
        self._maybe_run_exact_solver(timeLimit=timeLimit, numThreads=numThreads)

    def getResults(self):
        standalone = bool(self.pso_config["return_heuristic_result_without_cplex"])
        use_as_mip_start = bool(self.pso_config["use_as_mip_start"])
        if self.solverGurobi is not None and not standalone and use_as_mip_start:
            return self.solverGurobi.getResults()

        if self.global_best_solution is None:
            variables = empty_solution(self.problem)
            objective_value = float("inf")
        else:
            variables = self.global_best_solution
            objective_value = evaluate_solution_cost(self.problem, variables)

        return build_results_from_variables(
            self.problem,
            variables,
            objective_value,
            self.time,
        )

    def terminate(self):
        if self.solverGurobi is not None and hasattr(self.solverGurobi, "terminate"):
            self.solverGurobi.terminate()
