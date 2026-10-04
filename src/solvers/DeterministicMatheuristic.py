"""Deterministic TSP/restricted MPPRP/routing coordinator with optional handoff."""

import time

import numpy as np

from config import Config
from src.solvers._routing_common import (
    canonicalize_vehicles,
    compact_solution,
    materialize_routes,
    ordered_arcs,
    validate_variables,
)
from src.solvers._solver_common import (
    ProblemData,
    build_results_from_variables,
    objective_from_components,
    result_variables,
    solution_components,
)
from src.solvers.FeasibleParticleHeuristic import FeasibleParticleHeuristic
from src.solvers.MultProductProdctionRoutingProblem import (
    MultProductProdctionRoutingProblem,
)
from src.solvers.OrderedTsp import solve_tsp


class DeterministicMatheuristic:
    def __init__(self, map, dir, log):
        self.data, self.dir, self.log = map, dir, log
        self.problem = ProblemData.from_map(map)
        self.config = Config.snapshot()
        self.settings = self.config["solver"]["matheuristic"]
        self.stages, self.artifacts = [], []
        self.order, self.best, self.destination = None, None, None
        self.selected_stage, self.time = None, 0.0
        self.final_results = None
        self.destination_telemetry = {}
        self.destination_offset = 0.0

    def _score(self, variables):
        costs = solution_components(self.problem, variables)
        return (
            float(objective_from_components(self.problem, costs, self.config)),
            float(costs[3:].sum()),
        )

    def _constructive_start(self):
        heuristic = FeasibleParticleHeuristic(self.data, self.dir, self.log)
        compact = heuristic.build_population(np.zeros((1, heuristic.particle_dim)))[0]
        if not compact["feasible"]:
            return None
        rank = {i: r for r, i in enumerate(self.order)}
        compact["route_plan"] = [
            [sorted(route, key=rank.__getitem__) for route in period]
            for period in compact["route_plan"]
        ]
        values = materialize_routes(self.problem, compact)
        return (
            values
            if validate_variables(self.problem, values, ordered_arcs(self.order))[
                "feasible"
            ]
            else None
        )

    def _improvement_start(self, variables):
        compact = compact_solution(self.problem, variables)
        totals = variables["Q"].sum(axis=1)
        compact["route_plan"] = [
            [[i for i in route if totals[:, i, t].sum() > 0] for route in period]
            for t, period in enumerate(compact["route_plan"])
        ]
        return materialize_routes(self.problem, compact)

    def _stage(self, name, seed, limit, threads, profile, fixed=None):
        started = time.perf_counter()
        if seed is not None and self.config["solver"]["symmetry_breaking"]["hc1"]:
            seed = canonicalize_vehicles(self.problem, seed)
        stage = MultProductProdctionRoutingProblem(
            self.data,
            self.dir,
            self.log,
            {"start": seed is not None, "variables": seed},
            routing_profile=profile,
            fixed_plan=fixed,
            transport_objective=name == "route_improvement",
        )
        try:
            stage.solver(timeLimit=limit, numThreads=threads)
            extract_started = time.perf_counter()
            values = stage.get_snapshot() if stage.solCount else None
            extract_seconds = time.perf_counter() - extract_started
            report = (
                validate_variables(
                    self.problem,
                    values,
                    ordered_arcs(self.order) if name == "restricted" else None,
                )
                if values is not None
                else {"feasible": False, "violations": []}
            )
            if values is not None and fixed is not None:
                for key in ("X", "Y"):
                    if not np.allclose(values[key], fixed[key], atol=1e-6, rtol=0):
                        report["violations"].append(f"Plano fixado de {key} alterado")
                if not np.allclose(
                    values["Q"].sum(axis=1), fixed["delivery_totals"], atol=1e-6, rtol=0
                ):
                    report["violations"].append("Entregas agregadas fixadas alteradas")
                required = fixed["delivery_totals"].sum(axis=0) > 0
                visited = np.rint(values["eta"].sum(axis=0)) > 0
                if not np.array_equal(required[1:], visited[1:]):
                    report["violations"].append(
                        "Visitas não correspondem às entregas positivas"
                    )
                report["feasible"] = not report["violations"]
            has_incumbent = values is not None
            if has_incumbent:
                artifact = {
                    "stage": name,
                    "variables": {
                        k: v
                        for k, v in values.items()
                        if k == "eta" or self.settings["debug"]["enabled"]
                    },
                }
                if self.settings["debug"]["enabled"]:
                    artifact["variables"].update(stage.get_auxiliary_variables())
                self.artifacts.append(artifact)
            if not report["feasible"]:
                if values is not None:
                    self.log.warning(
                        f"{name}: incumbente rejeitada: {report['violations']}"
                    )
                values = None
            info = stage.model.solve_details
            record = {
                "stage": name,
                "status": str(info.status),
                "time_limit": limit,
                "solve_seconds": stage.time,
                "elapsed_seconds": time.perf_counter() - started,
                "build_seconds": getattr(stage, "build_seconds", None),
                "extract_seconds": extract_seconds,
                "variables": stage.model.number_of_variables,
                "constraints": stage.model.number_of_constraints,
                "bound": getattr(info, "best_bound", None),
                "gap": getattr(info, "mip_relative_gap", None),
                "gap_scope": name,
                "has_incumbent": has_incumbent,
                "has_solution": values is not None,
                "validation_errors": report["violations"],
            }
            record["features"] = getattr(stage, "applied_features", {})
            telemetry = stage.get_telemetry()
            record["rounded_capacity"] = telemetry.get("rounded_capacity", {})
            if values is not None:
                costs = solution_components(self.problem, values)
                record.update(
                    objective=float(stage.model.objective_value),
                    configured_objective=self._score(values)[0],
                    costs=costs.tolist(),
                )
                values = dict(values)
                for key in ("Z", "Y", "eta"):
                    if key in values:
                        values[key] = (
                            {
                                k: int(round(v))
                                for k, v in values[key].items()
                                if round(v)
                            }
                            if isinstance(values[key], dict)
                            else np.rint(values[key]).astype(int)
                        )
            record["elapsed_seconds"] = time.perf_counter() - started
            self.stages.append(record)
            return values
        finally:
            stage.terminate()

    def solver(self, numThreads=None, timeLimit=None):
        started = time.perf_counter()
        tsp = solve_tsp(
            self.problem.a_i_k, self.settings["tsp"]["time_limit"], numThreads, self.log
        )
        self.order = tsp["order"]
        self.stages.append(
            {
                "stage": "tsp",
                **{k: v for k, v in tsp.items() if k not in ("u", "order")},
                "time_limit": self.settings["tsp"]["time_limit"],
                "gap_scope": "tsp",
            }
        )
        self.artifacts.append({"stage": "tsp", "variables": {"u": tsp["u"]}})
        self.best = self._stage(
            "restricted",
            self._constructive_start(),
            self.settings["restricted"]["time_limit"],
            numThreads,
            {"mode": "ordered", "order": self.order},
        )
        if self.best is not None:
            self.selected_stage = "restricted"
            if self.settings["route_improvement"]["enabled"]:
                fixed = {
                    "X": self.best["X"],
                    "Y": self.best["Y"],
                    "delivery_totals": self.best["Q"].sum(axis=1),
                }
                candidate = self._stage(
                    "route_improvement",
                    self._improvement_start(self.best),
                    self.settings["route_improvement"]["time_limit"],
                    numThreads,
                    {"mode": "improvement"},
                    fixed,
                )
                if candidate is not None:
                    new, old = self._score(candidate), self._score(self.best)
                    if new[0] < old[0] - 1e-8 or (
                        abs(new[0] - old[0]) <= 1e-8 and new[1] < old[1] - 1e-8
                    ):
                        self.best, self.selected_stage = candidate, "route_improvement"
        use_as = self.settings["use_as"]
        if use_as != "final":
            if (
                self.best is not None
                and self.config["solver"]["symmetry_breaking"]["hc1"]
            ):
                self.best = canonicalize_vehicles(self.problem, self.best)
            handoff_started = time.perf_counter()
            self.destination_offset = handoff_started - started
            if use_as == "exact_start":
                self.destination = MultProductProdctionRoutingProblem(
                    self.data,
                    self.dir,
                    self.log,
                    {"start": self.best is not None, "variables": self.best},
                )
            else:
                from src.solvers.ParticleSwarmOptimization import (
                    ParticleSwarmOptimization,
                )

                self.destination = ParticleSwarmOptimization(
                    self.data, self.dir, self.log, initial_solution=self.best
                )
            try:
                self.destination.solver(timeLimit=timeLimit, numThreads=numThreads)
                result = self.destination.getResults()
                if hasattr(self.destination, "get_telemetry"):
                    telemetry = self.destination.get_telemetry()
                    if isinstance(telemetry, dict):
                        self.destination_telemetry = telemetry
                candidate_score, costs, report = (
                    None,
                    None,
                    {"feasible": False, "violations": []},
                )
                if result[11]:
                    # Every destination is audited before its final tuple is accepted.
                    values = result_variables(self.problem, result)
                    report = validate_variables(self.problem, values)
                    costs = solution_components(self.problem, values).tolist()
                    candidate_score = (
                        self._score(values)[0] if report["feasible"] else None
                    )
                    if report["feasible"] and (
                        self.best is None
                        or candidate_score <= self._score(self.best)[0] + 1e-8
                    ):
                        self.best, self.final_results, self.selected_stage = (
                            values,
                            result,
                            use_as,
                        )
                destination_model = getattr(self.destination, "model", None)
                if destination_model is None:
                    exact = getattr(self.destination, "solverGurobi", None)
                    destination_model = getattr(exact, "model", None)
                self.stages.append(
                    {
                        "stage": use_as,
                        "status": self.destination_telemetry.get("status", "completed"),
                        "elapsed_seconds": time.perf_counter() - handoff_started,
                        "time_limit": (
                            timeLimit
                            if use_as == "exact_start"
                            else self.config["solver"]["pso"]["time_limit"]
                        ),
                        "variables": getattr(
                            destination_model, "number_of_variables", None
                        ),
                        "constraints": getattr(
                            destination_model, "number_of_constraints", None
                        ),
                        "features": self.destination_telemetry.get("applied_features"),
                        "has_solution": bool(result[11]),
                        "selected": self.selected_stage == use_as,
                        "objective": result[7] if result[11] else None,
                        "configured_objective": candidate_score,
                        "costs": costs,
                        "validation_errors": report["violations"],
                        "bound": self.destination_telemetry.get("best_bound"),
                        "gap": self.destination_telemetry.get("relative_gap"),
                        "gap_scope": "unrestricted",
                    }
                )
            finally:
                self.destination.terminate()
                self.destination = None
        self.time = time.perf_counter() - started
        for stage in self.stages:
            stage["selected"] = stage["stage"] == self.selected_stage

    def getResults(self):
        if self.final_results is not None:
            result = list(self.final_results)
            result[9] = self.time
            return tuple(result)
        return build_results_from_variables(
            self.problem,
            self.best or {},
            self._score(self.best)[0] if self.best is not None else 0,
            self.time,
            config=self.config,
            solution_count=int(self.best is not None),
        )

    def get_artifacts(self):
        return {
            "stages": self.stages,
            "variables": self.artifacts,
            "metadata": {
                "order": self.order,
                "selected_stage": self.selected_stage,
                "total_seconds": self.time,
                "config": self.config,
                "instance_file": self.data.get("file"),
                "commit_hash": self.data.get("commit_hash"),
                "config_hash": self.data.get("config_hash"),
                "dimensions": {
                    "p": self.problem.p,
                    "v": self.problem.v,
                    "i": self.problem.i,
                    "t": self.problem.t,
                },
                "weight": self.data.get("weight"),
                "alpha": self.data.get("alpha"),
                "targets": self.data.get("targets"),
                "strengthened_bounds": self.data.get("strengthened_bounds", True),
                "sparse_zeroes": True,
                "debug": self.settings["debug"]["enabled"],
                "parameters": self.data if self.settings["debug"]["enabled"] else None,
            },
        }

    def get_telemetry(self):
        destination = dict(self.destination_telemetry)
        for key in ("first_feasible_seconds", "gap_target_seconds"):
            if destination.get(key) is not None:
                destination[key] += self.destination_offset
        events = [
            {
                **event,
                "elapsed_seconds": self.destination_offset + event["elapsed_seconds"],
            }
            for event in destination.get("mip_events", [])
        ]
        if self.selected_stage not in ("exact_start", "pso_start"):
            destination.update(
                best_bound=None,
                root_bound=None,
                relative_gap=None,
                gap_target_seconds=None,
                status="heuristic_complete" if self.best is not None else "no_solution",
            )
        return {
            "status": "heuristic_complete" if self.best is not None else "no_solution",
            "timed_out": False,
            "first_feasible_seconds": None,
            "gap_target_seconds": None,
            "mip_seconds": 0.0,
            "best_bound": None,
            "relative_gap": None,
            **destination,
            "strategy": "matheuristic_" + self.settings["use_as"],
            "total_seconds": self.time,
            "selected_stage": self.selected_stage,
            "objective": self._score(self.best)[0] if self.best is not None else None,
            "solution_count": int(self.best is not None),
            "mip_events": events,
            "pso_iterations": self.destination_telemetry.get("pso_iterations", []),
        }

    def terminate(self):
        if self.destination is not None:
            self.destination.terminate()
            self.destination = None
