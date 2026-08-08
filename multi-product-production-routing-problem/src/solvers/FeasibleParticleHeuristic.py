import math
import time

import numpy as np

from src.log.Logger import Logger
from src.solvers._solver_common import (
    ProblemData,
    clone_solution,
    empty_solution,
)


class FeasibleParticleHeuristic:
    def __init__(self, map, dir, log: Logger, rng=None, bounds=None, relaxed_base=None):
        self.data = map
        self.dir = dir
        self.log = log
        self.problem = ProblemData.from_map(map)
        self.rng = rng or np.random.default_rng(seed=123)
        self.dim_x = self.problem.p * self.problem.t
        self.dim_q = self.problem.p * self.problem.v * self.problem.i * self.problem.t
        self.particle_dim = self.dim_x + self.dim_q
        self.bounds = bounds
        self.relaxed_base = relaxed_base
        self.fast_validation_seconds = 0.0
        if bounds is not None:
            self.lower_bounds = np.asarray(bounds["lower"], dtype=float)
            self.upper_bounds = np.asarray(bounds["upper"], dtype=float)
            expected_shape = (self.problem.p, self.problem.t)
            if self.lower_bounds.shape != expected_shape or self.upper_bounds.shape != expected_shape:
                raise ValueError(f"bounds devem ter formato {expected_shape}")
            if np.any(self.lower_bounds > self.upper_bounds + 1e-8):
                raise ValueError("bounds relaxados inválidos: LB maior que UB")
            if "q_lower" in bounds or "q_upper" in bounds:
                self.lower_delivery_bounds = np.asarray(
                    bounds.get("q_lower"), dtype=float
                )
                self.upper_delivery_bounds = np.asarray(
                    bounds.get("q_upper"), dtype=float
                )
                expected_delivery_shape = (
                    self.problem.p,
                    self.problem.v,
                    self.problem.i,
                    self.problem.t,
                )
                if (
                    self.lower_delivery_bounds.shape != expected_delivery_shape
                    or self.upper_delivery_bounds.shape != expected_delivery_shape
                ):
                    raise ValueError(
                        f"bounds de Q devem ter formato {expected_delivery_shape}"
                    )
                if np.any(
                    self.lower_delivery_bounds > self.upper_delivery_bounds + 1e-8
                ):
                    raise ValueError("bounds relaxados de Q inválidos: LB maior que UB")
            else:
                self.lower_delivery_bounds = None
                self.upper_delivery_bounds = None
        else:
            self.lower_bounds = None
            self.upper_bounds = None
            self.lower_delivery_bounds = None
            self.upper_delivery_bounds = None

    def build_population(self, particles: np.ndarray):
        return [self._build_solution_from_particle(row) for row in particles]

    def repair_population(self, particles: np.ndarray, previous_solutions=None):
        repaired = []
        for index, row in enumerate(particles):
            solution = self._build_solution_from_particle(row)
            previous = (
                previous_solutions[index]
                if previous_solutions is not None and index < len(previous_solutions)
                else None
            )
            if not solution["feasible"] and previous is not None and previous["feasible"]:
                solution = clone_solution(previous)
                solution["reused_previous"] = True
            else:
                solution["reused_previous"] = False
            repaired.append(solution)
        return repaired

    def _empty_compact_solution(self):
        return {
            "X": np.zeros((self.problem.p, self.problem.t), dtype=int),
            "Y": np.zeros((self.problem.p, self.problem.t), dtype=int),
            "I": np.zeros(
                (self.problem.p, self.problem.i, self.problem.t), dtype=int
            ),
            "Q": np.zeros(
                (
                    self.problem.p,
                    self.problem.v,
                    self.problem.i,
                    self.problem.t,
                ),
                dtype=int,
            ),
            "route_plan": [],
            "assignments": [],
        }

    def validate_compact_solution(self, solution):
        """Fast vectorized validation used for every particle."""
        X = np.asarray(solution["X"])
        Y = np.asarray(solution["Y"])
        I = np.asarray(solution["I"])
        Q = np.asarray(solution["Q"])
        violations = []

        if np.any(X < 0):
            violations.append("X negativo")
        if np.any((Y != 0) & (Y != 1)):
            violations.append("Y nao binario")
        if np.any(X > self.problem.M * Y):
            violations.append("X excede M*Y")

        production_time = np.asarray(self.problem.b_p, dtype=int) @ X
        if np.any(production_time > self.problem.B):
            violations.append("capacidade producao excedida")

        inventory_capacity = np.asarray(self.problem.U_p_i, dtype=int)[:, :, None]
        if np.any(I < 0):
            violations.append("estoque negativo")
        if np.any(I > inventory_capacity):
            violations.append("capacidade estoque excedida")

        initial_inventory = np.asarray(self.problem.I_p_i_0, dtype=int)
        previous_plant = np.concatenate(
            (initial_inventory[:, 0, None], I[:, 0, :-1]), axis=1
        )
        delivered_from_plant = Q[:, :, 1:, :].sum(axis=(1, 2))
        if np.any(previous_plant + X - delivered_from_plant != I[:, 0, :]):
            violations.append("balanco planta")

        previous_customers = np.concatenate(
            (initial_inventory[:, 1:, None], I[:, 1:, :-1]), axis=2
        )
        received = Q[:, :, 1:, :].sum(axis=1)
        demand = np.asarray(self.problem.d_p_i_t, dtype=int)
        if np.any(previous_customers + received - demand != I[:, 1:, :]):
            violations.append("balanco cliente")

        vehicle_load = Q[:, :, 1:, :].sum(axis=(0, 2))
        if np.any(vehicle_load > self.problem.C):
            violations.append("capacidade veiculo")

        active_vehicle = Q[:, :, 1:, :].sum(axis=0) > 0
        if np.any(active_vehicle.sum(axis=0) > 1):
            violations.append("cliente atendido por mais de um veiculo")

        route_plan = solution.get("route_plan", [])
        if len(route_plan) != self.problem.t:
            violations.append("plano de rotas incompleto")
        else:
            for t, period_routes in enumerate(route_plan):
                if len(period_routes) != self.problem.v:
                    violations.append(f"quantidade de rotas invalida t={t}")
                    continue
                for vehicle, route in enumerate(period_routes):
                    route = list(route)
                    if len(route) != len(set(route)):
                        violations.append(
                            f"cliente repetido v={vehicle} t={t}"
                        )
                    expected = set(
                        np.flatnonzero(active_vehicle[vehicle, :, t]) + 1
                    )
                    if set(route) != expected:
                        violations.append(
                            f"rota inconsistente v={vehicle} t={t}"
                        )

        return {"feasible": not violations, "violations": violations[:50]}

    def materialize_solution(self, solution):
        """Create dense R/Z arrays only for audits, output and warm starts."""
        full = empty_solution(self.problem)
        for name in ("X", "Y", "I", "Q"):
            full[name][...] = solution[name]
        for t, routes in enumerate(solution.get("route_plan", [])):
            self._fill_route_variables(full, t, routes)
        metadata = clone_solution(
            {
                "route_plan": solution.get("route_plan", []),
                "assignments": solution.get("assignments", []),
            }
        )
        return {
            **full,
            **metadata,
            "routes": solution.get("routes", []),
            "visits": solution.get("visits", []),
            "feasible": solution.get("feasible", False),
            "cost": solution.get("cost", math.inf),
        }

    def validate_solution(self, solution):
        """Complete vectorized audit, including materialized route flow."""
        compact_report = self.validate_compact_solution(solution)
        if not compact_report["feasible"]:
            return compact_report
        full = solution if "R" in solution and "Z" in solution else self.materialize_solution(solution)
        route_capacity = np.asarray(full["R"]).sum(axis=0)
        route_enabled_capacity = self.problem.C * np.asarray(full["Z"])
        violations = list(compact_report["violations"])
        if np.any(route_capacity > route_enabled_capacity):
            violations.append("capacidade rota")
        return {"feasible": not violations, "violations": violations[:50]}

    def evaluate_compact_cost(self, solution):
        X = np.asarray(solution["X"], dtype=int)
        Y = np.asarray(solution["Y"], dtype=int)
        I = np.asarray(solution["I"], dtype=int)
        total = int(np.sum(np.asarray(self.problem.s_p)[:, None] * Y))
        total += int(np.sum(np.asarray(self.problem.c_p)[:, None] * X))
        total += int(
            np.sum(np.asarray(self.problem.h_p_i)[:, :, None] * I)
        )
        for period_routes in solution.get("route_plan", []):
            for route in period_routes:
                if not route:
                    continue
                total += self.problem.f
                full_route = [0] + list(route) + [0]
                total += sum(
                    self.problem.a_i_k[full_route[index]][full_route[index + 1]]
                    for index in range(len(full_route) - 1)
                )
        return total

    def _build_solution_from_particle(self, row):
        if row.shape[0] != self.particle_dim:
            raise ValueError(
                f"particle_dim invalido: esperado {self.particle_dim}, recebido {row.shape[0]}"
            )

        raw_x, raw_q = self._decode_particle(row)
        solution = self._empty_compact_solution()
        previous_inventory = np.array(self.problem.I_p_i_0, dtype=int)

        for t in range(self.problem.t):
            period_state = self._build_period_state(t, raw_x, raw_q, previous_inventory)
            if period_state is None:
                infeasible = self._empty_compact_solution()
                return {
                    **infeasible,
                    "routes": [],
                    "visits": [],
                    "feasible": False,
                    "cost": math.inf,
                }

            production, period_deliveries, vehicle_assignment, routes = period_state
            solution["route_plan"].append([list(route) for route in routes])
            solution["assignments"].append(dict(vehicle_assignment))

            for p in range(self.problem.p):
                solution["X"][p, t] = production[p]
                solution["Y"][p, t] = 1 if production[p] > 0 else 0

            for customer, assigned_vehicle in vehicle_assignment.items():
                for p in range(self.problem.p):
                    qty = period_deliveries[customer][p]
                    solution["Q"][p, assigned_vehicle, customer, t] = qty

            for p in range(self.problem.p):
                delivered_from_plant = sum(
                    solution["Q"][p, v, i, t]
                    for v in range(self.problem.v)
                    for i in range(1, self.problem.i)
                )
                solution["I"][p, 0, t] = previous_inventory[p, 0] + solution["X"][p, t] - delivered_from_plant

                for i in range(1, self.problem.i):
                    delivered_to_customer = sum(
                        solution["Q"][p, v, i, t] for v in range(self.problem.v)
                    )
                    solution["I"][p, i, t] = (
                        previous_inventory[p, i]
                        + delivered_to_customer
                        - self.problem.d_p_i_t[p][i - 1][t]
                    )

            previous_inventory = solution["I"][:, :, t].copy()

        validation_started_at = time.perf_counter()
        report = self.validate_compact_solution(solution)
        self.fast_validation_seconds += time.perf_counter() - validation_started_at
        cost = self.evaluate_compact_cost(solution) if report["feasible"] else math.inf
        return {
            **clone_solution(solution),
            "routes": [
                {"periodo": t, "route": [list(route) for route in routes]}
                for t, routes in enumerate(solution["route_plan"])
            ],
            "visits": self._extract_visits(solution),
            "feasible": report["feasible"],
            "cost": cost,
        }

    def _decode_particle(self, row):
        raw_x = row[: self.dim_x].reshape(self.problem.p, self.problem.t)
        raw_q = row[self.dim_x :].reshape(
            self.problem.p, self.problem.v, self.problem.i, self.problem.t
        )
        return raw_x, raw_q

    def _normalize_gene(self, value):
        if np.isnan(value):
            return 0.5
        bounded_value = min(60.0, max(-60.0, float(value)))
        return 1.0 / (1.0 + math.exp(-bounded_value))

    def _pick_int_in_range(self, low, high, gene):
        low = int(max(0, round(low)))
        high = int(max(low, round(high)))
        if high == low:
            return low
        normalized = self._normalize_gene(gene)
        return low + int(round(normalized * (high - low)))

    def _bounded_production_from_gene(self, p, t, gene):
        if self.lower_bounds is None:
            return None
        normalized = self._normalize_gene(gene)
        value = self.lower_bounds[p, t] + normalized * (
            self.upper_bounds[p, t] - self.lower_bounds[p, t]
        )
        return int(round(value))

    def _bounded_delivery_from_genes(self, p, customer, t, raw_q):
        if self.lower_delivery_bounds is None:
            return None
        total = 0.0
        for v in range(self.problem.v):
            lower = self.lower_delivery_bounds[p, v, customer, t]
            upper = self.upper_delivery_bounds[p, v, customer, t]
            total += lower + self._normalize_gene(raw_q[p, v, customer, t]) * (
                upper - lower
            )
        return int(round(total))

    def _build_period_state(self, t, raw_x, raw_q, previous_inventory):
        deficits = {
            customer: [
                max(0, int(self.problem.d_p_i_t[p][customer - 1][t] - previous_inventory[p, customer]))
                for p in range(self.problem.p)
            ]
            for customer in range(1, self.problem.i)
        }

        minimum_production = np.zeros(self.problem.p, dtype=int)
        for p in range(self.problem.p):
            total_deficit = sum(deficits[customer][p] for customer in deficits)
            minimum_production[p] = max(0, total_deficit - int(previous_inventory[p, 0]))

        minimum_time = sum(
            self.problem.b_p[p] * minimum_production[p] for p in range(self.problem.p)
        )
        if minimum_time > self.problem.B:
            return None

        production = minimum_production.copy()
        if self.lower_bounds is not None:
            production = np.maximum(
                production,
                np.ceil(self.lower_bounds[:, t]).astype(int),
            )
        minimum_time = sum(
            self.problem.b_p[p] * production[p] for p in range(self.problem.p)
        )
        if minimum_time > self.problem.B:
            return None
        remaining_time = int(self.problem.B - minimum_time)
        priority_products = sorted(
            range(self.problem.p),
            key=lambda p: self._normalize_gene(raw_x[p, t]),
            reverse=True,
        )

        for p in priority_products:
            if self.problem.b_p[p] <= 0:
                continue
            max_extra_by_time = int(remaining_time // self.problem.b_p[p])
            if self.lower_bounds is None:
                storage_headroom = max(
                    0,
                    int(self.problem.U_p_i[p][0] - previous_inventory[p, 0]),
                )
                extra_high = min(storage_headroom, max_extra_by_time)
            else:
                decoded = self._bounded_production_from_gene(p, t, raw_x[p, t])
                bound_extra = max(0, decoded - int(production[p]))
                extra_high = min(max_extra_by_time, bound_extra)
            if extra_high <= 0:
                continue
            if self.lower_bounds is None:
                extra_units = self._pick_int_in_range(0, extra_high, raw_x[p, t])
            else:
                extra_units = extra_high
            production[p] += extra_units
            remaining_time -= self.problem.b_p[p] * extra_units

        period_deliveries = {customer: [0 for _ in range(self.problem.p)] for customer in range(1, self.problem.i)}

        for customer in range(1, self.problem.i):
            lower_by_product = deficits[customer]
            lower_total = sum(lower_by_product)
            if lower_total > self.problem.C:
                return None

            for p in range(self.problem.p):
                period_deliveries[customer][p] = lower_by_product[p]

        vehicle_assignment = self._assign_customers_to_vehicles(
            period_deliveries,
            raw_q,
            t,
        )
        if vehicle_assignment is None:
            return None

        if not self._drain_plant_inventory(
            period_deliveries,
            previous_inventory,
            production,
            raw_q,
            t,
            vehicle_assignment,
        ):
            return None

        routes = self._build_nearest_neighbor_routes(vehicle_assignment)
        return production, period_deliveries, vehicle_assignment, routes

    def _drain_plant_inventory(
        self,
        period_deliveries,
        previous_inventory,
        production,
        raw_q,
        t,
        vehicle_assignment,
    ):
        available_by_product = np.array(
            previous_inventory[:, 0] + production,
            dtype=int,
        )
        mandatory_by_product = np.array(
            [
                sum(period_deliveries[customer][p] for customer in period_deliveries)
                for p in range(self.problem.p)
            ],
            dtype=int,
        )
        plant_inventory_before_extra = available_by_product - mandatory_by_product
        if np.any(plant_inventory_before_extra < 0):
            return False

        # A planta pode carregar estoque para o próximo período.  Portanto,
        # somente o volume acima de U[p][0] precisa ser entregue agora.
        excess_by_product = np.maximum(
            0,
            plant_inventory_before_extra
            - np.asarray(self.problem.U_p_i, dtype=int)[:, 0],
        )

        remaining_capacity = [self.problem.C for _ in range(self.problem.v)]
        for customer, vehicle in vehicle_assignment.items():
            remaining_capacity[vehicle] -= sum(period_deliveries[customer])
        if any(capacity < 0 for capacity in remaining_capacity):
            return False

        product_priority = sorted(
            range(self.problem.p),
            key=lambda p: sum(
                max(
                    0,
                    self.problem.U_p_i[p][customer]
                    - (
                        previous_inventory[p, customer]
                        + period_deliveries[customer][p]
                        - self.problem.d_p_i_t[p][customer - 1][t]
                    ),
                )
                for customer in period_deliveries
            ),
        )
        for p in product_priority:
            while excess_by_product[p] > 0:
                candidates = []
                for customer in period_deliveries:
                    customer_inventory = (
                        previous_inventory[p, customer]
                        + period_deliveries[customer][p]
                        - self.problem.d_p_i_t[p][customer - 1][t]
                    )
                    stock_headroom = int(
                        self.problem.U_p_i[p][customer] - customer_inventory
                    )
                    if stock_headroom <= 0:
                        continue

                    if customer in vehicle_assignment:
                        vehicles = [vehicle_assignment[customer]]
                    else:
                        vehicles = sorted(
                            range(self.problem.v),
                            key=lambda vehicle: self._normalize_gene(
                                raw_q[p, vehicle, customer, t]
                            ),
                            reverse=True,
                        )
                    for vehicle in vehicles:
                        available = min(stock_headroom, remaining_capacity[vehicle])
                        if available <= 0:
                            continue
                        preference = self._normalize_gene(
                            raw_q[p, vehicle, customer, t]
                        )
                        if self.lower_delivery_bounds is not None:
                            desired = self._bounded_delivery_from_genes(
                                p,
                                customer,
                                t,
                                raw_q,
                            )
                            preference += int(
                                desired > period_deliveries[customer][p]
                            )
                        candidates.append(
                            (preference, -remaining_capacity[vehicle], customer, vehicle, available)
                        )

                if not candidates:
                    return False

                _, _, customer, vehicle, available = max(candidates)
                quantity = min(int(excess_by_product[p]), available)
                period_deliveries[customer][p] += quantity
                remaining_capacity[vehicle] -= quantity
                vehicle_assignment[customer] = vehicle
                excess_by_product[p] -= quantity
        return True

    def _assign_customers_to_vehicles(self, period_deliveries, raw_q, t):
        remaining_capacity = [self.problem.C for _ in range(self.problem.v)]
        assignment = {}
        customers = sorted(
            [customer for customer in period_deliveries if sum(period_deliveries[customer]) > 0],
            key=lambda customer: sum(period_deliveries[customer]),
            reverse=True,
        )

        for customer in customers:
            total_delivery = sum(period_deliveries[customer])
            vehicle_preferences = sorted(
                range(self.problem.v),
                key=lambda vehicle: self._normalize_gene(
                    sum(raw_q[p, vehicle, customer, t] for p in range(self.problem.p))
                ),
                reverse=True,
            )
            feasible_vehicles = [
                vehicle
                for vehicle in vehicle_preferences
                if remaining_capacity[vehicle] >= total_delivery
            ]
            if not feasible_vehicles:
                return None
            vehicle = min(
                feasible_vehicles,
                key=lambda candidate: remaining_capacity[candidate] - total_delivery,
            )
            assignment[customer] = vehicle
            remaining_capacity[vehicle] -= total_delivery

        return assignment

    def _build_nearest_neighbor_routes(self, vehicle_assignment):
        routes = [[] for _ in range(self.problem.v)]
        vehicle_customers = {vehicle: [] for vehicle in range(self.problem.v)}
        for customer, vehicle in vehicle_assignment.items():
            vehicle_customers[vehicle].append(customer)

        for vehicle, customers in vehicle_customers.items():
            if not customers:
                continue

            unvisited = set(customers)
            current = 0
            route = []
            while unvisited:
                next_customer = min(
                    unvisited,
                    key=lambda customer: self.problem.a_i_k[current][customer],
                )
                route.append(next_customer)
                unvisited.remove(next_customer)
                current = next_customer
            routes[vehicle] = route

        return routes

    def _fill_route_variables(self, solution, t, routes):
        for vehicle, route in enumerate(routes):
            if not route:
                continue

            full_route = [0] + route + [0]
            for index in range(len(full_route) - 1):
                origin = full_route[index]
                destination = full_route[index + 1]
                solution["Z"][vehicle, origin, destination, t] = 1
                remaining_customers = full_route[index + 1 : -1]
                for p in range(self.problem.p):
                    solution["R"][p, vehicle, origin, destination, t] = sum(
                        solution["Q"][p, vehicle, customer, t]
                        for customer in remaining_customers
                    )

    def _extract_routes_metadata(self, solution):
        routes = []
        for t in range(self.problem.t):
            period_routes = []
            for vehicle in range(self.problem.v):
                current = 0
                route = []
                visited = set()
                while True:
                    next_nodes = [
                        destination
                        for destination in range(self.problem.k)
                        if solution["Z"][vehicle, current, destination, t] == 1
                    ]
                    if not next_nodes:
                        break
                    destination = next_nodes[0]
                    route.append(destination)
                    if destination == 0 or destination in visited:
                        break
                    visited.add(destination)
                    current = destination
                period_routes.append(route[:-1] if route and route[-1] == 0 else route)
            routes.append({"periodo": t, "route": period_routes})
        return routes

    def _extract_visits(self, solution):
        visits = []
        for t in range(self.problem.t):
            period_visits = []
            for customer in range(1, self.problem.i):
                total_delivery = sum(
                    solution["Q"][p, v, customer, t]
                    for p in range(self.problem.p)
                    for v in range(self.problem.v)
                )
                period_visits.append(1 if total_delivery > 0 else 0)
            visits.append(period_visits)
        return visits
