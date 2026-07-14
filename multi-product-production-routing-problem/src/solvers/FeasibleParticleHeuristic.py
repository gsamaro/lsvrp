import math

import numpy as np

from src.log.Logger import Logger
from src.solvers._solver_common import (
    ProblemData,
    clone_solution,
    empty_solution,
    evaluate_solution_cost,
)


class FeasibleParticleHeuristic:
    def __init__(self, map, dir, log: Logger, rng=None):
        self.data = map
        self.dir = dir
        self.log = log
        self.problem = ProblemData.from_map(map)
        self.rng = rng or np.random.default_rng(seed=123)
        self.dim_x = self.problem.p * self.problem.t
        self.dim_q = self.problem.p * self.problem.v * self.problem.i * self.problem.t
        self.particle_dim = self.dim_x + self.dim_q

    def build_population(self, particles: np.ndarray):
        return [self._build_solution_from_particle(row) for row in particles]

    def repair_population(self, particles: np.ndarray, previous_solutions=None):
        return [self._build_solution_from_particle(row) for row in particles]

    def validate_solution(self, solution):
        variables = solution
        X = variables["X"]
        Y = variables["Y"]
        I = variables["I"]
        Q = variables["Q"]
        R = variables["R"]
        Z = variables["Z"]
        violations = []

        for p in range(self.problem.p):
            for t in range(self.problem.t):
                if X[p, t] < 0:
                    violations.append(f"X[{p},{t}] negativo")
                if Y[p, t] not in (0, 1):
                    violations.append(f"Y[{p},{t}] nao binario")
                if X[p, t] > self.problem.M * Y[p, t]:
                    violations.append(f"X[{p},{t}] excede M*Y")

        for t in range(self.problem.t):
            total_time = sum(self.problem.b_p[p] * X[p, t] for p in range(self.problem.p))
            if total_time > self.problem.B:
                violations.append(f"capacidade producao excedida t={t}")

        for p in range(self.problem.p):
            for i in range(self.problem.i):
                for t in range(self.problem.t):
                    if I[p, i, t] < 0:
                        violations.append(f"I[{p},{i},{t}] negativo")
                    if I[p, i, t] > self.problem.U_p_i[p][i]:
                        violations.append(f"I[{p},{i},{t}] excede U")

        for p in range(self.problem.p):
            for t in range(self.problem.t):
                previous = self.problem.I_p_i_0[p][0] if t == 0 else I[p, 0, t - 1]
                delivered = sum(Q[p, v, i, t] for v in range(self.problem.v) for i in range(1, self.problem.i))
                if previous + X[p, t] - delivered != I[p, 0, t]:
                    violations.append(f"balanco planta p={p} t={t}")

        for p in range(self.problem.p):
            for i in range(1, self.problem.i):
                for t in range(self.problem.t):
                    previous = self.problem.I_p_i_0[p][i] if t == 0 else I[p, i, t - 1]
                    delivered = sum(Q[p, v, i, t] for v in range(self.problem.v))
                    if previous + delivered - self.problem.d_p_i_t[p][i - 1][t] != I[p, i, t]:
                        violations.append(f"balanco cliente p={p} i={i} t={t}")

        for v in range(self.problem.v):
            for i in range(self.problem.i):
                for k in range(self.problem.k):
                    if i == k:
                        continue
                    for t in range(self.problem.t):
                        if sum(R[p, v, i, k, t] for p in range(self.problem.p)) > self.problem.C * Z[v, i, k, t]:
                            violations.append(f"capacidade rota v={v} i={i} k={k} t={t}")

        return {"feasible": len(violations) == 0, "violations": violations[:50]}

    def _build_solution_from_particle(self, row):
        if row.shape[0] != self.particle_dim:
            raise ValueError(
                f"particle_dim invalido: esperado {self.particle_dim}, recebido {row.shape[0]}"
            )

        raw_x, raw_q = self._decode_particle(row)
        solution = empty_solution(self.problem)
        previous_inventory = np.array(self.problem.I_p_i_0, dtype=int)

        for t in range(self.problem.t):
            period_state = self._build_period_state(t, raw_x, raw_q, previous_inventory)
            if period_state is None:
                infeasible = empty_solution(self.problem)
                return {
                    **infeasible,
                    "routes": [],
                    "visits": [],
                    "feasible": False,
                    "cost": math.inf,
                }

            production, period_deliveries, vehicle_assignment, routes = period_state

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

            self._fill_route_variables(solution, t, routes)
            previous_inventory = solution["I"][:, :, t].copy()

        cost = evaluate_solution_cost(self.problem, solution)
        report = self.validate_solution(solution)
        return {
            **clone_solution(solution),
            "routes": self._extract_routes_metadata(solution),
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
        return 1.0 / (1.0 + math.exp(-float(value)))

    def _pick_int_in_range(self, low, high, gene):
        low = int(max(0, round(low)))
        high = int(max(low, round(high)))
        if high == low:
            return low
        normalized = self._normalize_gene(gene)
        return low + int(round(normalized * (high - low)))

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
        remaining_time = int(self.problem.B - minimum_time)
        priority_products = sorted(
            range(self.problem.p),
            key=lambda p: self._normalize_gene(raw_x[p, t]),
            reverse=True,
        )

        for p in priority_products:
            if self.problem.b_p[p] <= 0:
                continue
            storage_headroom = max(
                0,
                int(self.problem.U_p_i[p][0] - previous_inventory[p, 0]),
            )
            max_extra_by_time = int(remaining_time // self.problem.b_p[p])
            extra_high = min(storage_headroom, max_extra_by_time)
            if extra_high <= 0:
                continue
            extra_units = self._pick_int_in_range(0, extra_high, raw_x[p, t])
            production[p] += extra_units
            remaining_time -= self.problem.b_p[p] * extra_units

        available_plant = np.array(previous_inventory[:, 0] + production, dtype=int)
        remaining_total_vehicle_capacity = self.problem.v * self.problem.C
        period_deliveries = {customer: [0 for _ in range(self.problem.p)] for customer in range(1, self.problem.i)}

        for customer in range(1, self.problem.i):
            lower_by_product = deficits[customer]
            lower_total = sum(lower_by_product)
            remaining_lower_totals_rest = sum(
                sum(deficits[other]) for other in range(customer + 1, self.problem.i)
            )
            max_total_for_customer = min(
                self.problem.C,
                remaining_total_vehicle_capacity - remaining_lower_totals_rest,
            )
            if max_total_for_customer < lower_total:
                return None

            remaining_customer_capacity = max_total_for_customer
            for p in range(self.problem.p):
                demand = int(self.problem.d_p_i_t[p][customer - 1][t])
                prev_customer_inventory = int(previous_inventory[p, customer])
                lower = lower_by_product[p]
                if prev_customer_inventory >= demand:
                    lower = 0

                remaining_required_for_rest = sum(
                    deficits[other][p] for other in range(customer + 1, self.problem.i)
                )
                if t == self.problem.t - 1:
                    upper = lower
                else:
                    stock_upper = max(
                        lower,
                        int(self.problem.U_p_i[p][customer] - prev_customer_inventory),
                    )
                    plant_upper = max(
                        lower,
                        int(available_plant[p] - remaining_required_for_rest),
                    )
                    upper = min(stock_upper, plant_upper, remaining_customer_capacity)
                    upper = max(lower, upper)

                gene = sum(raw_q[p, v, customer, t] for v in range(self.problem.v))
                delivery = self._pick_int_in_range(lower, upper, gene)
                period_deliveries[customer][p] = delivery
                available_plant[p] -= delivery
                remaining_customer_capacity -= delivery

            delivered_total = sum(period_deliveries[customer])
            remaining_total_vehicle_capacity -= delivered_total

        vehicle_assignment = self._assign_customers_to_vehicles(period_deliveries, raw_q, t)
        if vehicle_assignment is None:
            return None

        routes = self._build_nearest_neighbor_routes(vehicle_assignment)
        return production, period_deliveries, vehicle_assignment, routes

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
            assigned = False
            for vehicle in vehicle_preferences:
                if remaining_capacity[vehicle] >= total_delivery:
                    assignment[customer] = vehicle
                    remaining_capacity[vehicle] -= total_delivery
                    assigned = True
                    break

            if not assigned:
                for vehicle in range(self.problem.v):
                    if remaining_capacity[vehicle] >= total_delivery:
                        assignment[customer] = vehicle
                        remaining_capacity[vehicle] -= total_delivery
                        assigned = True
                        break

            if not assigned:
                return None

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
