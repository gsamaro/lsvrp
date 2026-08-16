import hashlib
import math
import os
import time

import numpy as np

from config import Config
from src.log.Logger import Logger
from src.helpers.SolverTelemetry import get_config
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
from src.solvers._pso_numba_kernel import update_swarm_kernel


class ParticleSwarmOptimization:
    def __init__(self, map, dir, log: Logger):
        self.data = map
        self.dir = dir
        self.log = log
        self.problem = ProblemData.from_map(map)
        self.pso_config = self._load_pso_config()
        self.telemetry_config = get_config(Config)
        self._pso_first_feasible_seconds = None
        self.bounds_seconds = 0.0
        self.jit_warmup_seconds = 0.0
        self.initialization_seconds = 0.0
        self.pso_hot_elapsed_seconds = 0.0
        self.pso_cold_elapsed_seconds = 0.0
        self.parallel_workers = self._resolve_parallel_workers(
            self.pso_config["parallel_workers"]
        )
        self.relaxed_bounds = None
        self.relaxed_base = None
        self.relaxed_total_bounds = None
        bounds_config = self.pso_config["lot_sizing_bounds"]
        if bounds_config["enabled"]:
            bounds_started_at = time.perf_counter()
            self.log.debug(">> Calculando bounds relaxados de dimensionamento de lotes.")
            time_limit = bounds_config["time_limit"]
            if time_limit is None:
                time_limit = Config.get_nested("solver", "timeLimit", default=None)
            relaxation = LotSizingRelaxation(
                map=map,
                log=log,
                time_limit=time_limit,
                integer_variables=bounds_config.get("integer_variables", True),
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
            self.log.debug(
                f">> Bounds agregados do PL: lower={relaxed['lower']} "
                f"upper={relaxed['upper']} "
                f"lower_gap={relaxed['lower_gap']} "
                f"upper_gap={relaxed['upper_gap']}"
            )
            self.bounds_seconds = time.perf_counter() - bounds_started_at
        self.heuristic = FeasibleParticleHeuristic(
            map=map,
            dir=dir,
            log=log,
            bounds=self.relaxed_bounds,
            relaxed_base=self.relaxed_base,
            execution_backend=self.pso_config["execution_backend"],
            parallel_workers=self.parallel_workers,
        )
        self.rng = np.random.default_rng(seed=self.pso_config["seed"])
        self.audit_rng = np.random.default_rng(seed=int(self.pso_config["seed"]) + 1)
        self.positions = None
        self.velocities = None
        self.solutions = []
        self.population_state = None
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
        self._stagnation_count = 0
        self._best_first_seen_iteration = None
        self._current_iteration = 0
        self._mutated_particles = 0
        self._reinitialized_particles = 0
        self._audits_executed = 0
        self._audit_failures = 0
        self._phase_times = self._new_phase_times()
        self._random_states = None
        self._adapter_seconds_accounted = 0.0
        self.log.debug(
            f"PSO execution backend={self.pso_config['execution_backend']} "
            f"parallel_workers={self.parallel_workers}"
        )
        if self.pso_config["execution_backend"] == "numba" and self.pso_config["jit_warmup"]:
            self.jit_warmup_seconds = self.heuristic.warmup_numba()

    def _load_pso_config(self):
        configured = Config.get_nested("solver", "pso", default={}) or {}
        defaults = {
            "swarm_size": 20,
            "max_iterations": 50,
            "inertia": 0.7,
            "cognitive": 1.5,
            "social": 1.5,
            "inertia_initial": 0.9,
            "inertia_final": 0.4,
            "cognitive_initial": 1.8,
            "cognitive_final": 1.2,
            "social_initial": 0.6,
            "social_final": 1.8,
            "velocity_limit": 2.0,
            "mutation_particle_rate": 0.10,
            "mutation_x_gene_rate": 0.10,
            "mutation_q_gene_rate": 0.005,
            "mutation_sigma_initial": 0.5,
            "mutation_sigma_final": 0.1,
            "stagnation_patience": 5,
            "reinitialize_fraction": 0.15,
            "elite_fraction": 0.05,
            "audit_fraction": 0.01,
            "audit_interval": 10,
            "execution_backend": "numba",
            "parallel_workers": "auto",
            "jit_warmup": True,
            "seed": 123,
            "repair_on_update": True,
            "max_initial_resample_attempts": 20,
            "lot_sizing_bounds": {
                "enabled": True,
                "use_base_solution": False,
                "time_limit": None,
                "integer_variables": True,
            },
            "use_as_mip_start": True,
            "return_heuristic_result_without_cplex": False,
        }
        merged = {**defaults, **configured}
        merged["lot_sizing_bounds"] = {
            **defaults["lot_sizing_bounds"],
            **(configured.get("lot_sizing_bounds") or {}),
        }
        for legacy_key, initial_key, final_key in (
            ("inertia", "inertia_initial", "inertia_final"),
            ("cognitive", "cognitive_initial", "cognitive_final"),
            ("social", "social_initial", "social_final"),
        ):
            if (
                legacy_key in configured
                and initial_key not in configured
                and final_key not in configured
            ):
                merged[initial_key] = configured[legacy_key]
                merged[final_key] = configured[legacy_key]
        return merged

    @staticmethod
    def _running_under_mpi():
        for name in (
            "OMPI_COMM_WORLD_SIZE",
            "PMI_SIZE",
            "SLURM_NTASKS",
            "PBS_NP",
        ):
            value = os.environ.get(name)
            if value:
                try:
                    if int(value) > 1:
                        return True
                except ValueError:
                    continue
        return False

    def _resolve_parallel_workers(self, configured):
        if isinstance(configured, str) and configured.strip().lower() == "auto":
            if self._running_under_mpi():
                return 1
            return max(1, min(8, os.cpu_count() or 1))
        return max(1, int(configured))

    @staticmethod
    def _new_phase_times():
        return {
            "update": 0.0,
            "build": 0.0,
            "kernel": 0.0,
            "adapter": 0.0,
            "fast_validation": 0.0,
            "audit": 0.0,
            "materialization": 0.0,
            "telemetry": 0.0,
        }

    def _random_positions(self):
        return self.rng.normal(
            loc=0.0,
            scale=1.0,
            size=(int(self.pso_config["swarm_size"]), self.heuristic.particle_dim),
        ).astype(np.float32)

    def _initialize_swarm(self):
        initialization_started_at = time.perf_counter()
        self._phase_times = self._new_phase_times()
        self.positions = self._random_positions()
        self.velocities = np.zeros_like(self.positions)
        seed_rng = np.random.default_rng(int(self.pso_config["seed"]) + 2)
        self._random_states = seed_rng.integers(
            1,
            np.iinfo(np.uint64).max,
            size=len(self.positions),
            dtype=np.uint64,
        )
        self._seed_relaxed_base_position()
        kernel_before = self.heuristic.kernel_seconds
        started_at = time.perf_counter()
        self.population_state = self.heuristic.build_population_state(self.positions)
        build_elapsed = time.perf_counter() - started_at
        kernel_elapsed = self.heuristic.kernel_seconds - kernel_before
        self._phase_times["kernel"] += kernel_elapsed
        self._phase_times["build"] += max(0.0, build_elapsed - kernel_elapsed)
        self._repair_invalid_particles(allow_resample=True)
        self._initialize_bests()
        diversity = self._record_population_metrics(iteration=0)
        self.initialization_seconds = time.perf_counter() - initialization_started_at
        if diversity["feasible"] and self._pso_first_feasible_seconds is None:
            self._pso_first_feasible_seconds = self.initialization_seconds
        self.log.debug(
            "PSO initialization "
            f"feasible={diversity['feasible']}/{len(self.population_state)} "
            f"x_profiles={diversity['x_profiles']} "
            f"q_profiles={diversity['q_profiles']} "
            f"q_quantity_profiles={diversity['q_quantity_profiles']} "
            f"vehicle_assignment_profiles={diversity['vehicle_assignment_profiles']} "
            f"q_full_profiles={diversity['q_full_profiles']}"
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

    def _repair_invalid_particles(self, allow_resample=False):
        invalid_indices = np.flatnonzero(self.population_state.feasible == 0)
        self._invalid_before_repair.append(len(invalid_indices))
        if not allow_resample:
            return
        for index in invalid_indices:
            replacement = self._resample_solution()
            if replacement is None:
                continue
            self.positions[index] = replacement["position"]
            self._copy_state_row(replacement["state"], 0, self.population_state, int(index))

    @staticmethod
    def _copy_state_row(source, source_index, destination, destination_index):
        for name in (
            "X",
            "Y",
            "I",
            "Q",
            "assignments",
            "route_nodes",
            "route_lengths",
            "feasible",
            "costs",
            "x_fingerprints",
            "q_quantity_fingerprints",
            "assignment_fingerprints",
            "q_full_fingerprints",
            "reused_previous",
        ):
            getattr(destination, name)[destination_index] = getattr(source, name)[source_index]

    def _resample_solution(self):
        for _ in range(int(self.pso_config["max_initial_resample_attempts"])):
            self._resample_attempts += 1
            candidate_position = self.rng.normal(
                loc=0.0,
                scale=1.0,
                size=(self.heuristic.particle_dim,),
            )
            candidate_position = candidate_position.astype(np.float32)
            candidate_state = self.heuristic.build_population_state(
                candidate_position[None, :]
            )
            if candidate_state.feasible[0]:
                return {"position": candidate_position, "state": candidate_state}
            self._resample_infeasible += 1
        return None

    def _initialize_bests(self):
        self.personal_best_positions = np.array(self.positions, copy=True)
        self.personal_best_costs = np.where(
            self.population_state.feasible,
            self.population_state.costs,
            float("inf"),
        )
        self.personal_best_solutions = []

        for best_index in np.argsort(self.personal_best_costs):
            if not np.isfinite(self.personal_best_costs[best_index]):
                break
            candidate = self.heuristic.solution_from_state(
                self.population_state, int(best_index)
            )
            if self._audit_solution(candidate):
                self.global_best_position = np.array(
                    self.personal_best_positions[best_index], copy=True
                )
                self.global_best_solution = clone_solution(candidate)
                self.global_best_cost = float(self.personal_best_costs[best_index])
                self._best_first_seen_iteration = 0
                return
            self.personal_best_costs[best_index] = float("inf")

    def _schedule_value(self, initial_key, final_key, iteration):
        total_iterations = max(1, int(self.pso_config["max_iterations"]) - 1)
        progress = min(1.0, max(0.0, (iteration - 1) / total_iterations))
        initial = float(self.pso_config[initial_key])
        final = float(self.pso_config[final_key])
        return initial + progress * (final - initial)

    def _elite_indices(self):
        population_size = len(self.personal_best_costs)
        count = max(
            1,
            int(math.ceil(float(self.pso_config["elite_fraction"]) * population_size)),
        )
        return set(np.argsort(self.personal_best_costs)[:count].tolist())

    def _mutate_non_elites(self, iteration):
        self._mutated_particles = 0
        population_size = len(self.positions)
        count = int(
            math.ceil(
                float(self.pso_config["mutation_particle_rate"])
                * population_size
            )
        )
        if count <= 0:
            return
        elite = self._elite_indices()
        candidates = np.array(
            [index for index in range(population_size) if index not in elite],
            dtype=int,
        )
        if not len(candidates):
            return
        selected = self.rng.choice(
            candidates, size=min(count, len(candidates)), replace=False
        )
        sigma = self._schedule_value(
            "mutation_sigma_initial", "mutation_sigma_final", iteration
        )
        x_gene_count = max(
            1,
            int(
                round(
                    float(self.pso_config["mutation_x_gene_rate"])
                    * self.heuristic.dim_x
                )
            ),
        )
        q_gene_count = max(
            1,
            int(
                round(
                    float(self.pso_config["mutation_q_gene_rate"])
                    * self.heuristic.dim_q
                )
            ),
        )
        for index in selected:
            x_indices = self.rng.choice(
                self.heuristic.dim_x,
                size=min(x_gene_count, self.heuristic.dim_x),
                replace=False,
            )
            q_indices = self.heuristic.dim_x + self.rng.choice(
                self.heuristic.dim_q,
                size=min(q_gene_count, self.heuristic.dim_q),
                replace=False,
            )
            gene_indices = np.concatenate((x_indices, q_indices))
            self.positions[index, gene_indices] += self.rng.normal(
                0.0, sigma, size=len(gene_indices)
            )
        self._mutated_particles = len(selected)

    def _reinitialize_if_stagnant(self):
        self._reinitialized_particles = 0
        patience = int(self.pso_config["stagnation_patience"])
        if patience <= 0 or self._stagnation_count < patience:
            return
        population_size = len(self.positions)
        count = int(
            math.ceil(
                float(self.pso_config["reinitialize_fraction"])
                * population_size
            )
        )
        elite = self._elite_indices()
        worst_first = np.argsort(self.personal_best_costs)[::-1]
        selected = [index for index in worst_first if index not in elite][:count]
        if not selected:
            return
        selected = np.asarray(selected, dtype=int)
        self.positions[selected] = self.rng.normal(
            0.0, 1.0, size=(len(selected), self.heuristic.particle_dim)
        )
        self.velocities[selected] = 0.0
        self.personal_best_positions[selected] = self.positions[selected]
        self.personal_best_costs[selected] = float("inf")
        if self.population_state is not None:
            self.population_state.feasible[selected] = 0
            self.population_state.costs[selected] = float("inf")
        elif self.solutions:
            for index in selected:
                self.solutions[int(index)] = {"feasible": False}
        self._reinitialized_particles = len(selected)
        self._stagnation_count = 0

    def _update_positions(self, iteration=1):
        started_at = time.perf_counter()
        inertia = self._schedule_value("inertia_initial", "inertia_final", iteration)
        cognitive = self._schedule_value(
            "cognitive_initial", "cognitive_final", iteration
        )
        social = self._schedule_value("social_initial", "social_final", iteration)
        social_target = (
            self.global_best_position
            if self.global_best_position is not None
            else self.positions
        )
        velocity_limit = float(self.pso_config["velocity_limit"])
        if self.pso_config["execution_backend"] == "numba" and self._random_states is not None:
            update_swarm_kernel(
                self.positions,
                self.velocities,
                self.personal_best_positions,
                np.asarray(social_target, dtype=np.float32),
                self._random_states,
                inertia,
                cognitive,
                social,
                velocity_limit,
            )
        else:
            r1 = self.rng.random(size=self.positions.shape)
            r2 = self.rng.random(size=self.positions.shape)
            self.velocities = (
                inertia * self.velocities
                + cognitive * r1 * (self.personal_best_positions - self.positions)
                + social * r2 * (social_target - self.positions)
            )
            np.clip(self.velocities, -velocity_limit, velocity_limit, out=self.velocities)
            self.positions = self.positions + self.velocities
        self._mutate_non_elites(iteration)
        self._reinitialize_if_stagnant()
        self._phase_times["update"] += time.perf_counter() - started_at

    def _audit_solution(self, solution):
        materialization_started_at = time.perf_counter()
        full = self.heuristic.materialize_solution(solution)
        self._phase_times["materialization"] += (
            time.perf_counter() - materialization_started_at
        )
        audit_started_at = time.perf_counter()
        report = self.heuristic.validate_solution(full)
        self._phase_times["audit"] += time.perf_counter() - audit_started_at
        self._audits_executed += 1
        if not report["feasible"]:
            self._audit_failures += 1
            self.log.error(
                "PSO audit failure violations=" + ", ".join(report["violations"])
            )
        return report["feasible"]

    def _audit_population(self, iteration):
        interval = int(self.pso_config["audit_interval"])
        if interval <= 0 or iteration % interval != 0:
            return
        feasible_indices = np.flatnonzero(self.population_state.feasible)
        if not len(feasible_indices):
            return
        sample_size = max(
            1,
            int(
                math.ceil(
                    float(self.pso_config["audit_fraction"])
                    * len(feasible_indices)
                )
            ),
        )
        selected = self.audit_rng.choice(
            feasible_indices,
            size=min(sample_size, len(feasible_indices)),
            replace=False,
        )
        for index in selected:
            solution = self.heuristic.solution_from_state(
                self.population_state, int(index)
            )
            if not self._audit_solution(solution):
                self.population_state.feasible[int(index)] = 0
                self.population_state.costs[int(index)] = float("inf")

    def _evaluate_swarm(self, iteration=1):
        previous_state = self.population_state
        kernel_before = self.heuristic.kernel_seconds
        build_started_at = time.perf_counter()
        candidate_state = self.heuristic.build_population_state(self.positions)
        invalid_indices = np.flatnonzero(candidate_state.feasible == 0)
        self._invalid_before_repair.append(len(invalid_indices))
        for index in invalid_indices:
            if previous_state is not None and previous_state.feasible[index]:
                self._copy_state_row(previous_state, int(index), candidate_state, int(index))
                candidate_state.reused_previous[index] = True
                self.positions[index] -= self.velocities[index]
                self.velocities[index] = 0
        self.population_state = candidate_state
        build_elapsed = time.perf_counter() - build_started_at
        kernel_elapsed = self.heuristic.kernel_seconds - kernel_before
        self._phase_times["kernel"] += kernel_elapsed
        self._phase_times["build"] += max(0.0, build_elapsed - kernel_elapsed)
        self._audit_population(iteration)

        costs = np.where(
            self.population_state.feasible,
            self.population_state.costs,
            float("inf"),
        )
        improved_personal = costs < self.personal_best_costs
        self.personal_best_costs[improved_personal] = costs[improved_personal]
        self.personal_best_positions[improved_personal] = self.positions[improved_personal]

        improved = False
        for best_index in np.argsort(self.personal_best_costs):
            candidate_cost = float(self.personal_best_costs[best_index])
            if not np.isfinite(candidate_cost) or candidate_cost >= self.global_best_cost:
                break
            if (
                self.population_state.feasible[best_index]
                and self.population_state.costs[best_index] == candidate_cost
            ):
                candidate = self.heuristic.solution_from_state(
                    self.population_state, int(best_index)
                )
            else:
                candidate_state = self.heuristic.build_population_state(
                    self.personal_best_positions[int(best_index)][None, :]
                )
                candidate = self.heuristic.solution_from_state(candidate_state, 0)
            if candidate["feasible"] and self._audit_solution(candidate):
                self.global_best_cost = candidate_cost
                self.global_best_position = np.array(
                    self.personal_best_positions[best_index], copy=True
                )
                self.global_best_solution = clone_solution(candidate)
                self._best_first_seen_iteration = iteration
                improved = True
                break
            self.personal_best_costs[best_index] = float("inf")
        self._stagnation_count = 0 if improved else self._stagnation_count + 1

    def _population_diversity(self):
        feasible_indices = np.flatnonzero(self.population_state.feasible)
        x_profiles = np.unique(
            self.population_state.x_fingerprints[feasible_indices]
        )
        q_quantity_profiles = np.unique(
            self.population_state.q_quantity_fingerprints[feasible_indices]
        )
        vehicle_assignment_profiles = np.unique(
            self.population_state.assignment_fingerprints[feasible_indices]
        )
        q_full_profiles = np.unique(
            self.population_state.q_full_fingerprints[feasible_indices]
        )
        costs = self.population_state.costs[feasible_indices]
        return {
            "feasible": len(feasible_indices),
            "reused_previous": int(self.population_state.reused_previous.sum()),
            "x_profiles": len(x_profiles),
            # Mantém q_profiles como alias compatível para o perfil de
            # quantidades agregadas por cliente.
            "q_profiles": len(q_quantity_profiles),
            "q_quantity_profiles": len(q_quantity_profiles),
            "vehicle_assignment_profiles": len(vehicle_assignment_profiles),
            "q_full_profiles": len(q_full_profiles),
            "cost_mean": float(np.mean(costs)) if len(costs) else float("inf"),
            "cost_max": float(np.max(costs)) if len(costs) else float("inf"),
            "cost_std": float(np.std(costs)) if len(costs) else float("inf"),
            "cost_unique": int(len(np.unique(costs))) if len(costs) else 0,
        }

    @staticmethod
    def _fingerprint(array):
        contiguous = np.ascontiguousarray(array)
        return hashlib.blake2b(contiguous.view(np.uint8), digest_size=8).digest()

    def _record_population_metrics(self, iteration):
        started_at = time.perf_counter()
        adapter_delta = (
            self.heuristic.adapter_seconds - self._adapter_seconds_accounted
        )
        self._phase_times["adapter"] += max(0.0, adapter_delta)
        self._adapter_seconds_accounted = self.heuristic.adapter_seconds
        metrics = {
            "iteration": iteration,
            "population_size": len(self.population_state),
            "best_cost": self.global_best_cost,
            **self._population_diversity(),
            "mutated_particles": self._mutated_particles,
            "reinitialized_particles": self._reinitialized_particles,
            "stagnation_count": self._stagnation_count,
            "audits_executed": self._audits_executed,
            "audit_failures": self._audit_failures,
        }
        self._phase_times["telemetry"] += time.perf_counter() - started_at
        metrics["phase_times"] = dict(self._phase_times)
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
        minimum_q_quantity_profiles = min(
            metrics["q_quantity_profiles"]
            for metrics in self._population_history
        )
        minimum_vehicle_assignment_profiles = min(
            metrics["vehicle_assignment_profiles"]
            for metrics in self._population_history
        )
        minimum_q_full_profiles = min(
            metrics["q_full_profiles"] for metrics in self._population_history
        )
        self.log.info(
            "PSO final "
            f"best_cost={best_cost} "
            f"best_first_seen_iteration={first_best_iteration} "
            f"final_feasible={final['feasible']}/{final['population_size']}"
        )
        self.log.debug(
            "PSO final diagnostics "
            f"pre_repair_infeasible={invalid_before_repair}/{total_population_candidates} "
            f"resample_infeasible={self._resample_infeasible}/{self._resample_attempts} "
            f"reused_previous={sum(metrics['reused_previous'] for metrics in self._population_history)} "
            f"x_profiles={initial['x_profiles']}->{final['x_profiles']}(min={minimum_x_profiles}) "
            f"q_profiles={initial['q_profiles']}->{final['q_profiles']}(min={minimum_q_profiles}) "
            f"q_quantity_profiles={initial['q_quantity_profiles']}->{final['q_quantity_profiles']}(min={minimum_q_quantity_profiles}) "
            f"vehicle_assignment_profiles={initial['vehicle_assignment_profiles']}->{final['vehicle_assignment_profiles']}(min={minimum_vehicle_assignment_profiles}) "
            f"q_full_profiles={initial['q_full_profiles']}->{final['q_full_profiles']}(min={minimum_q_full_profiles}) "
            f"audits={self._audits_executed} audit_failures={self._audit_failures} "
            f"mutated={sum(metrics['mutated_particles'] for metrics in self._population_history)} "
            f"reinitialized={sum(metrics['reinitialized_particles'] for metrics in self._population_history)}"
        )

    def _materialized_global_best(self):
        if self.global_best_solution is None:
            return None
        return self.heuristic.materialize_solution(self.global_best_solution)

    def _maybe_run_exact_solver(self, numThreads=None, timeLimit=None, full_best=None):
        standalone = bool(self.pso_config["return_heuristic_result_without_cplex"])
        use_as_mip_start = bool(self.pso_config["use_as_mip_start"])

        if standalone or not use_as_mip_start:
            return

        if full_best is None:
            full_best = self._materialized_global_best()
        if full_best is None:
            start = {"start": False}
        else:
            start = {
                "start": True,
                "variables": {
                    "X": full_best["X"],
                    "Y": full_best["Y"],
                    "I": full_best["I"],
                    "Q": full_best["Q"],
                    "R": full_best["R"],
                    "Z": full_best["Z"],
                },
            }

        self.solverGurobi = MPPRP(self.data, self.dir, self.log, start)
        self.solverGurobi.solver(timeLimit=timeLimit, numThreads=numThreads)

    def solver(self, numThreads=None, timeLimit=None):
        started_at = time.perf_counter()
        self._initialize_swarm()

        for iteration in range(int(self.pso_config["max_iterations"])):
            iteration_number = iteration + 1
            self._current_iteration = iteration_number
            self._phase_times = self._new_phase_times()
            iteration_started_at = time.perf_counter()
            self._update_positions(iteration=iteration_number)
            self._evaluate_swarm(iteration=iteration_number)
            diversity = self._record_population_metrics(iteration=iteration_number)
            diversity["elapsed_seconds"] = (
                time.perf_counter() - iteration_started_at
            )
            self.log.debug(
                f"PSO iteration={iteration_number} "
                f"best_cost={self.global_best_cost} "
                f"feasible={diversity['feasible']}/{len(self.population_state)} "
                f"reused_previous={diversity['reused_previous']} "
                f"x_profiles={diversity['x_profiles']} "
                f"q_profiles={diversity['q_profiles']} "
                f"q_quantity_profiles={diversity['q_quantity_profiles']} "
                f"vehicle_assignment_profiles={diversity['vehicle_assignment_profiles']} "
                f"q_full_profiles={diversity['q_full_profiles']} "
                f"cost_mean={diversity['cost_mean']:.2f} "
                f"cost_max={diversity['cost_max']:.2f} "
                f"cost_std={diversity['cost_std']:.2f} "
                f"cost_unique={diversity['cost_unique']} "
                f"mutated={diversity['mutated_particles']} "
                f"reinitialized={diversity['reinitialized_particles']} "
                f"stagnation={diversity['stagnation_count']} "
                f"audits={diversity['audits_executed']} "
                f"audit_failures={diversity['audit_failures']} "
                f"phase_times={diversity['phase_times']} "
                f"elapsed={diversity['elapsed_seconds']:.4f}s"
            )

        self.time = time.perf_counter() - started_at
        self.solCount = 1 if self.global_best_solution is not None else 0
        if self.global_best_solution is not None and not self._audit_solution(
            self.global_best_solution
        ):
            raise RuntimeError("melhor solução do PSO falhou na auditoria final")
        self._log_final_report()
        full_best = self._materialized_global_best()
        self.pso_hot_elapsed_seconds = self.bounds_seconds + (
            time.perf_counter() - started_at
        )
        self.pso_cold_elapsed_seconds = (
            self.pso_hot_elapsed_seconds + self.jit_warmup_seconds
        )
        self.log.debug(
            "PSO timing "
            f"bounds_seconds={self.bounds_seconds:.6f} "
            f"jit_warmup_seconds={self.jit_warmup_seconds:.6f} "
            f"initialization_seconds={self.initialization_seconds:.6f} "
            f"pso_hot_elapsed_seconds={self.pso_hot_elapsed_seconds:.6f} "
            f"pso_cold_elapsed_seconds={self.pso_cold_elapsed_seconds:.6f}"
        )
        self._maybe_run_exact_solver(
            timeLimit=timeLimit,
            numThreads=numThreads,
            full_best=full_best,
        )

    def get_telemetry(self):
        if self.solverGurobi is not None and hasattr(self.solverGurobi, "get_telemetry"):
            exact = self.solverGurobi.get_telemetry()
            pso_elapsed = float(self.pso_hot_elapsed_seconds)
            events = []
            if self._pso_first_feasible_seconds is not None:
                events.append({"event": "first_feasible", "elapsed_seconds": self.bounds_seconds + self._pso_first_feasible_seconds, "source": "pso"})
            for event in exact.get("mip_events", []):
                item = dict(event)
                item["elapsed_seconds"] = pso_elapsed + float(item["elapsed_seconds"])
                events.append(item)
            first = (self.bounds_seconds + self._pso_first_feasible_seconds if self._pso_first_feasible_seconds is not None else None)
            exact_first = exact.get("first_feasible_seconds")
            if exact_first is not None:
                exact_first += pso_elapsed
                first = exact_first if first is None else min(first, exact_first)
            gap_target = exact.get("gap_target_seconds")
            if gap_target is not None:
                gap_target += pso_elapsed
            return {**exact, "strategy": "pso_mip_start", "pso_hot_seconds": pso_elapsed,
                    "pso_cold_seconds": float(self.pso_cold_elapsed_seconds),
                    "bounds_seconds": float(self.bounds_seconds), "jit_warmup_seconds": float(self.jit_warmup_seconds),
                    "initialization_seconds": float(self.initialization_seconds),
                    "first_feasible_seconds": first, "gap_target_seconds": gap_target,
                    "mip_events": events, "pso_iterations": list(self._population_history)}
        return {"strategy": "pso_standalone", "mip_seconds": 0.0, "status": "heuristic_complete",
                "timed_out": False, "objective": self.global_best_cost, "best_bound": None,
                "relative_gap": None, "node_count": 0, "solution_count": self.solCount,
                "first_feasible_seconds": (self.bounds_seconds + self._pso_first_feasible_seconds if self._pso_first_feasible_seconds is not None else None),
                "gap_target_seconds": None,
                "mip_events": ([{"event": "first_feasible", "elapsed_seconds": self.bounds_seconds + self._pso_first_feasible_seconds, "source": "pso"}] if self._pso_first_feasible_seconds is not None else []),
                "pso_hot_seconds": float(self.pso_hot_elapsed_seconds), "pso_cold_seconds": float(self.pso_cold_elapsed_seconds),
                "bounds_seconds": float(self.bounds_seconds), "jit_warmup_seconds": float(self.jit_warmup_seconds),
                "initialization_seconds": float(self.initialization_seconds), "pso_iterations": list(self._population_history)}

    def getResults(self):
        standalone = bool(self.pso_config["return_heuristic_result_without_cplex"])
        use_as_mip_start = bool(self.pso_config["use_as_mip_start"])
        if self.solverGurobi is not None and not standalone and use_as_mip_start:
            return self.solverGurobi.getResults()

        if self.global_best_solution is None:
            variables = empty_solution(self.problem)
            objective_value = float("inf")
        else:
            variables = self._materialized_global_best()
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
