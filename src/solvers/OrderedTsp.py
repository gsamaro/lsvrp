"""Time-limited symmetric TSP with a deterministic valid fallback and LRU cache."""

import hashlib
import time
from collections import OrderedDict

import numpy as np
from docplex.mp.model import Model

_CACHE = OrderedDict()


def tour_cost(order, distance):
    return float(sum(distance[i, j] for i, j in zip(order, order[1:])))


def initial_tour(distance):
    remaining, order = set(range(1, len(distance))), [0]
    while remaining:
        customer = min(remaining, key=lambda i: (distance[order[-1], i], i))
        order.append(customer)
        remaining.remove(customer)
    order.append(0)
    improved = True
    while improved:
        improved = False
        for a in range(1, len(order) - 2):
            for b in range(a + 1, len(order) - 1):
                delta = (
                    distance[order[a - 1], order[b]]
                    + distance[order[a], order[b + 1]]
                    - distance[order[a - 1], order[a]]
                    - distance[order[b], order[b + 1]]
                )
                if delta < -1e-9:
                    order[a : b + 1] = reversed(order[a : b + 1])
                    improved = True
    reverse = [0, *reversed(order[1:-1]), 0]
    return min(order, reverse)


def successor_cycles(arcs, node_count):
    outgoing = dict(arcs)
    if len(outgoing) != node_count or set(outgoing.values()) != set(range(node_count)):
        raise ValueError("Solução TSP não é uma cobertura por ciclos")
    unseen, cycles = set(range(node_count)), []
    while unseen:
        first = min(unseen)
        cycle, node = [], first
        while node in unseen:
            unseen.remove(node)
            cycle.append(node)
            node = outgoing[node]
        if node != first:
            raise ValueError("Cobertura TSP inválida")
        cycles.append(cycle)
    return cycles


def build_tsp_model(distance):
    model = Model(name="MPPRP_Order_TSP")
    count = len(distance)
    u = model.binary_var_dict(
        ((i, j) for i in range(count) for j in range(count) if i != j), name="u"
    )
    model.minimize(
        model.sum(distance[i, j] * variable for (i, j), variable in u.items())
    )
    for i in range(count):
        model.add_constraint(model.sum(u[i, j] for j in range(count) if i != j) == 1)
        model.add_constraint(model.sum(u[j, i] for j in range(count) if i != j) == 1)
    return model, u


def subtour_rows(arcs, node_count, indices):
    """SEC rows for a candidate cycle cover; usable without a CPLEX callback."""
    return [
        ([indices[i, j] for i in cycle for j in cycle if i != j], len(cycle) - 1)
        for cycle in successor_cycles(arcs, node_count)
        if len(cycle) < node_count
    ]


def solve_tsp(distance, time_limit, threads, log):
    distance = np.ascontiguousarray(distance, dtype=float)
    if (
        distance.ndim != 2
        or distance.shape[0] != distance.shape[1]
        or not np.all(np.isfinite(distance))
        or not np.allclose(distance, distance.T)
    ):
        raise ValueError("TSP requer matriz de custos finita, quadrada e simétrica")
    key = (
        hashlib.sha256(distance.tobytes()).hexdigest(),
        distance.shape,
        time_limit,
        threads,
    )
    if key in _CACHE:
        cached = _CACHE.pop(key)
        _CACHE[key] = cached
        return {
            **cached,
            "order": list(cached["order"]),
            "u": cached["u"].copy(),
            "cache_hit": True,
            "elapsed_seconds": 0.0,
            "build_seconds": 0.0,
            "solve_seconds": 0.0,
        }
    started = time.perf_counter()
    order = initial_tour(distance)
    summary = {
        "status": "constructive",
        "bound": None,
        "gap": None,
        "cache_hit": False,
        "solve_seconds": 0.0,
        "build_seconds": 0.0,
        "variables": 0,
        "constraints": 0,
    }
    if len(distance) > 1:
        model, u = build_tsp_model(distance)
        try:
            model.set_time_limit(time_limit)
            if threads is not None:
                model.parameters.threads = threads
            model.parameters.randomseed = 123
            warm = model.new_solution()
            selected = set(zip(order, order[1:]))
            for arc, variable in u.items():
                warm.add_var_value(variable, int(arc in selected))
            model.add_mip_start(warm)
            import cplex
            from cplex.callbacks import LazyConstraintCallback

            indices = {arc: var.index for arc, var in u.items()}
            count = len(distance)

            class SubtourCallback(LazyConstraintCallback):
                def __call__(self):
                    values = self.get_values(list(indices.values()))
                    arcs = [arc for arc, value in zip(indices, values) if value > 0.5]
                    for row, rhs in subtour_rows(arcs, count, indices):
                        self.add(
                            cplex.SparsePair(ind=row, val=[1.0] * len(row)), "L", rhs
                        )

            model.register_callback(SubtourCallback)
            summary["build_seconds"] = time.perf_counter() - started
            solve_started = time.perf_counter()
            solution = model.solve(log_output=False)
            summary["solve_seconds"] = time.perf_counter() - solve_started
            summary.update(
                status=str(model.solve_details.status),
                bound=getattr(model.solve_details, "best_bound", None),
                gap=getattr(model.solve_details, "mip_relative_gap", None),
                variables=model.number_of_variables,
                constraints=model.number_of_constraints,
            )
            if solution is not None:
                arcs = [arc for arc, var in u.items() if solution.get_value(var) > 0.5]
                cycles = successor_cycles(arcs, count)
                if len(cycles) == 1:
                    candidate = [*cycles[0], 0]
                    if (
                        tour_cost(candidate, distance)
                        <= tour_cost(order, distance) + 1e-8
                    ):
                        order = min(candidate, [0, *reversed(candidate[1:-1]), 0])
                else:
                    log.warning(
                        "Incumbente TSP com subtour rejeitada; mantendo rota inicial"
                    )
        finally:
            model.end()
    u_values = np.zeros_like(distance, dtype=int)
    for i, j in zip(order, order[1:]):
        if i != j:
            u_values[i, j] = 1
    result = {
        **summary,
        "order": order,
        "u": u_values,
        "objective": tour_cost(order, distance),
        "elapsed_seconds": time.perf_counter() - started,
    }
    _CACHE[key] = result
    if len(_CACHE) > 64:
        _CACHE.popitem(last=False)
    return {**result, "order": list(order), "u": u_values.copy()}
