import copy
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ProblemData:
    raw_map: dict
    p: int
    i: int
    k: int
    t: int
    v: int
    B: int
    b_p: list
    c_p: list
    s_p: list
    M: int
    U_p_i: list
    I_p_i_0: list
    h_p_i: list
    C: int
    f: float
    a_i_k: list
    d_p_i_t: list

    @classmethod
    def from_map(cls, map_data):
        return cls(
            raw_map=map_data,
            p=map_data["num_products"],
            i=map_data["num_customers"] + 1,
            k=map_data["num_customers"] + 1,
            t=map_data["num_periods"],
            v=map_data["num_vehicles"],
            B=map_data["B"],
            b_p=map_data["b_p"],
            c_p=map_data["c_p"],
            s_p=map_data["s_p"],
            M=map_data["M"],
            U_p_i=map_data["U_pi"],
            I_p_i_0=map_data["I_pi0"],
            h_p_i=map_data["h_pi"],
            C=map_data["C"],
            f=map_data["f"],
            a_i_k=map_data["a_ik"],
            d_p_i_t=map_data["d_pit"],
        )


def compute_production_upper_bounds(problem: ProblemData):
    """Return the strengthened upper bound for every ``(product, period)``.

    ``M_p`` is the product-specific replacement for the global big-M: the
    smaller of the instance big-M and the total demand of product ``p`` over
    the horizon.  ``D_remaining[p, t]`` is the demand of product ``p`` from
    period ``t`` onward.  The per-period bound is the minimum of these two
    quantities and the production-capacity bound ``B / b_p``.
    """

    demand = np.asarray(problem.d_p_i_t, dtype=float)
    total_demand = demand.sum(axis=(1, 2))
    product_big_m = np.minimum(float(problem.M), total_demand)
    bounds = np.zeros((problem.p, problem.t), dtype=float)

    for p in range(problem.p):
        production_capacity = (
            float(problem.B) / float(problem.b_p[p])
            if problem.b_p[p] > 0
            else product_big_m[p]
        )
        for t in range(problem.t):
            demand_remaining = float(demand[p, :, t:].sum())
            bounds[p, t] = max(
                0.0,
                min(product_big_m[p], production_capacity, demand_remaining),
            )
    return bounds


def compute_delivery_upper_bounds(problem: ProblemData):
    """Return ``qbar[p, customer, period]`` for customer deliveries."""

    result = np.zeros((problem.p, problem.i, problem.t), dtype=float)
    for p in range(problem.p):
        for i in range(1, problem.i):
            for t in range(problem.t):
                result[p, i, t] = min(
                    float(problem.C),
                    float(problem.U_p_i[p][i]) + float(problem.d_p_i_t[p][i - 1][t]),
                )
    return result


def compute_inventory_upper_bounds(problem: ProblemData):
    """Return tightened upper bounds for ``I[p, i, t]``.

    At a customer, at most one vehicle may visit per period, so cumulative
    receipts are bounded by one ``qbar`` per period.  The inventory balance
    also subtracts cumulative demand.  At the plant, cumulative production
    bounds the amount that can be added to the initial stock.  Existing
    capacity bounds remain valid and are always included.
    """

    production_bounds = compute_production_upper_bounds(problem)
    delivery_bounds = compute_delivery_upper_bounds(problem)
    demand = np.asarray(problem.d_p_i_t, dtype=float)
    result = np.zeros((problem.p, problem.i, problem.t), dtype=float)

    for p in range(problem.p):
        for i in range(problem.i):
            for t in range(problem.t):
                if i == 0:
                    reachable = float(problem.I_p_i_0[p][i]) + float(
                        production_bounds[p, : t + 1].sum()
                    )
                else:
                    net_receipts = (
                        delivery_bounds[p, i, : t + 1] - demand[p, i - 1, : t + 1]
                    )
                    reachable = float(problem.I_p_i_0[p][i]) + float(net_receipts.sum())
                result[p, i, t] = max(
                    0.0,
                    min(float(problem.U_p_i[p][i]), reachable),
                )
    return result


def empty_solution(problem: ProblemData):
    return {
        "X": np.zeros((problem.p, problem.t), dtype=float),
        "Y": np.zeros((problem.p, problem.t), dtype=int),
        "I": np.zeros((problem.p, problem.i, problem.t), dtype=float),
        "Q": np.zeros((problem.p, problem.v, problem.i, problem.t), dtype=float),
        "R": np.zeros(
            (problem.p, problem.v, problem.i, problem.k, problem.t), dtype=float
        ),
        "Z": np.zeros((problem.v, problem.i, problem.k, problem.t), dtype=int),
    }


def clone_solution(solution):
    cloned = {}
    for key, value in solution.items():
        if isinstance(value, np.ndarray):
            cloned[key] = np.array(value, copy=True)
        else:
            cloned[key] = copy.deepcopy(value)
    return cloned


def solution_components(problem: ProblemData, variables):
    """Return F[j,t], without materializing arc tensors for compact particles."""
    x, y, inventory = (np.asarray(variables[k], dtype=float) for k in ("X", "Y", "I"))
    result = np.zeros((5, problem.t), dtype=float)
    result[0] = np.asarray(problem.c_p) @ x
    result[1] = np.asarray(problem.s_p) @ y
    result[2] = np.einsum("pi,pit->t", np.asarray(problem.h_p_i), inventory)
    if "route_plan" in variables:
        for t, routes in enumerate(variables["route_plan"]):
            for route in routes:
                if route:
                    result[3, t] += problem.f
                    nodes = [0, *route, 0]
                    result[4, t] += sum(
                        problem.a_i_k[a][b] for a, b in zip(nodes, nodes[1:])
                    )
    else:
        if isinstance(variables["Z"], dict):
            for (v, i, j, t), value in variables["Z"].items():
                result[3, t] += problem.f * value if i == 0 and j != 0 else 0
                result[4, t] += problem.a_i_k[i][j] * value
            return result
        z = np.asarray(variables["Z"], dtype=float)
        if np.ndim(problem.f) == 0:
            result[3] = problem.f * z[:, 0, 1:, :].sum(axis=(0, 1))
        else:
            # Retain the historical writer's array-valued fixed-cost contract.
            result[3] = [
                sum(np.sum(problem.f * z[v, 0, :, t]) for v in range(z.shape[0]))
                for t in range(problem.t)
            ]
        result[4] = np.einsum("ij,vijt->t", np.asarray(problem.a_i_k), z)
    return result


def adjusted_targets(targets, periods):
    if targets is None or len(targets) != periods:
        raise ValueError("Metas multiobjetivo devem conter todos os períodos")
    original = np.asarray(
        [[targets[t][f"f{j+1}_target"] for t in range(periods)] for j in range(5)],
        dtype=float,
    )
    adjusted = np.where(original == 0, original.mean(axis=1)[:, None], original)
    if not np.all(np.isfinite(adjusted)) or np.any(adjusted <= 0):
        raise ValueError("Metas ajustadas devem ser positivas para normalização")
    return original, adjusted


def objective_from_components(problem, components, config):
    """Evaluate the same scalar criterion used by the configured exact model."""
    solver = config.get("solver", {})
    if config.get("postprocessing", {}).get("build_target", False):
        return np.einsum("j,...jt->...", problem.raw_map["weight"], components)
    if not solver.get("multiobjective", False):
        return np.sum(components, axis=(-2, -1))
    targets, normalizers = adjusted_targets(problem.raw_map.get("targets"), problem.t)
    positive = np.maximum(0.0, components - targets)
    weighted = positive * np.asarray(problem.raw_map["weight"])[:, None] / normalizers
    alpha = float(problem.raw_map["alpha"])
    return alpha * np.max(weighted, axis=(-2, -1)) + (1 - alpha) * np.sum(
        weighted, axis=(-2, -1)
    )


def evaluate_solution_cost(problem: ProblemData, variables):
    return float(solution_components(problem, variables).sum())


def evaluate_configured_objective(problem, variables, config):
    return float(
        objective_from_components(
            problem, solution_components(problem, variables), config
        )
    )


def goal_start_values(problem, variables):
    targets, normalizers = adjusted_targets(problem.raw_map.get("targets"), problem.t)
    difference = solution_components(problem, variables) - targets
    positive, negative = np.maximum(0, difference), np.maximum(0, -difference)
    limit = float(
        np.max(np.asarray(problem.raw_map["weight"])[:, None] * positive / normalizers)
    )
    return positive, negative, limit


def build_results_from_variables(
    problem: ProblemData,
    variables,
    objective_value,
    elapsed_time,
    config=None,
    solution_count=1,
):
    # The legacy tuple and tensor layouts are kept; quantities are never truncated.
    positive, limit, new_targets = [], None, None
    if (
        config
        and config.get("solver", {}).get("multiobjective", False)
        and solution_count
    ):
        p, _, limit = goal_start_values(problem, variables)
        positive = p.T.tolist()
        _, adjusted = adjusted_targets(problem.raw_map["targets"], problem.t)
        new_targets = {
            t: {f"f{j+1}_target": float(adjusted[j, t]) for j in range(5)}
            for t in range(problem.t)
        }
    if not solution_count:
        return ([], [], [], [], [], [], [], 0, 0, elapsed_time, None, 0, 0, 0, 0, None)
    from src.solvers._routing_common import materialize_snapshot

    variables = materialize_snapshot(problem, variables)
    return (
        np.rint(variables["Z"]).astype(int).transpose(3, 0, 1, 2).tolist(),
        np.asarray(variables["X"], dtype=float).T.tolist(),
        np.rint(variables["Y"]).astype(int).T.tolist(),
        np.asarray(variables["I"], dtype=float).transpose(2, 1, 0).tolist(),
        np.asarray(variables["R"], dtype=float).transpose(4, 1, 0, 2, 3).tolist(),
        np.asarray(variables["Q"], dtype=float).transpose(3, 1, 0, 2).tolist(),
        positive,
        objective_value,
        float("nan"),
        elapsed_time,
        limit,
        solution_count,
        float("nan"),
        0,
        float("nan"),
        new_targets,
    )


def result_variables(problem, result):
    return {
        "Z": np.asarray(result[0], dtype=float).transpose(1, 2, 3, 0),
        "X": np.asarray(result[1], dtype=float).T,
        "Y": np.asarray(result[2], dtype=float).T,
        "I": np.asarray(result[3], dtype=float).transpose(2, 1, 0),
        "R": np.asarray(result[4], dtype=float).transpose(2, 1, 3, 4, 0),
        "Q": np.asarray(result[5], dtype=float).transpose(2, 1, 3, 0),
    }
