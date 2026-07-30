import time

import numpy as np

from config import Config
from src.log.Logger import Logger
from src.solvers.FeasibleParticleHeuristic import FeasibleParticleHeuristic
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
        self.heuristic = FeasibleParticleHeuristic(map=map, dir=dir, log=log)
        self.pso_config = self._load_pso_config()
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
            "use_as_mip_start": True,
            "return_heuristic_result_without_cplex": False,
        }
        return {**defaults, **configured}

    def _random_positions(self):
        return self.rng.normal(
            loc=0.0,
            scale=1.0,
            size=(int(self.pso_config["swarm_size"]), self.heuristic.particle_dim),
        )

    def _initialize_swarm(self):
        self.positions = self._random_positions()
        self.velocities = np.zeros_like(self.positions)
        self.solutions = self.heuristic.build_population(self.positions)
        self._repair_invalid_particles()
        self._initialize_bests()

    def _repair_invalid_particles(self):
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
            invalid = sum(1 for solution in self.solutions if not solution["feasible"])
            self.log.info(
                f"PSO iteration={iteration + 1} "
                f"best_cost={self.global_best_cost} "
                f"invalid_particles={invalid} "
                f"elapsed={time.time() - iteration_started_at:.2f}s"
            )

        self.time = time.time() - started_at
        self.solCount = 1 if self.global_best_solution is not None else 0
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
