"""Numeric routing audits and sparse snapshots shared by stages and seeds."""

import numpy as np

from src.solvers._solver_common import clone_solution, empty_solution


class ZeroArcDict(dict):
    """Omitted arcs are structural zeroes, never newly allocated variables."""

    def __missing__(self, key):
        return 0


def ordered_arcs(order):
    customers = list(order[1:-1])
    if (
        order[0] != 0
        or order[-1] != 0
        or len(customers) != len(set(customers))
        or 0 in customers
    ):
        raise ValueError(
            "Ordem deve iniciar/terminar na planta e conter clientes distintos"
        )
    return tuple(
        [(0, i) for i in customers]
        + [(i, j) for r, i in enumerate(customers) for j in customers[r + 1 :]]
        + [(i, 0) for i in customers]
    )


def arc_items(values):
    if isinstance(values, dict):
        return values.items()
    array = np.asarray(values)
    return ((tuple(key), float(array[tuple(key)])) for key in np.argwhere(array != 0))


def routes_from_z(z, tolerance=1e-6, shape=None):
    """Read dense output or a sparse {(v,i,k,t):value} snapshot."""
    shape = shape if isinstance(z, dict) else np.asarray(z).shape
    if shape is None or len(shape) != 4 or shape[1] != shape[2]:
        raise ValueError("Dimensões de Z inválidas")
    by_route = [[set() for _ in range(shape[0])] for _ in range(shape[3])]
    for (v, i, j, t), value in arc_items(z):
        if not (
            0 <= v < shape[0]
            and 0 <= i < shape[1]
            and 0 <= j < shape[2]
            and 0 <= t < shape[3]
        ):
            raise ValueError("Índice de arco inválido")
        if (
            not np.isfinite(value)
            or abs(value - round(value)) > tolerance
            or value < -tolerance
            or value > 1 + tolerance
        ):
            raise ValueError("Arcos não binários ou fora de [0,1]")
        if value > 0.5:
            by_route[t][v].add((i, j))
    result = []
    for period_arcs in by_route:
        period = []
        for arcs in period_arcs:
            if any(i == j for i, j in arcs):
                raise ValueError("Arco próprio")
            outgoing, incoming = {}, {}
            for i, j in arcs:
                if i in outgoing or j in incoming:
                    raise ValueError("Grau de rota maior que um")
                outgoing[i], incoming[j] = j, i
            route, seen, current = [], set(), 0
            while current in outgoing:
                arc = (current, outgoing[current])
                if arc in seen:
                    raise ValueError("Ciclo inválido")
                seen.add(arc)
                current = arc[1]
                if current == 0:
                    break
                route.append(current)
            if seen != arcs or (arcs and current != 0):
                raise ValueError("Rota desconectada da planta")
            period.append(route)
        visited = [i for route in period for i in route]
        if len(visited) != len(set(visited)):
            raise ValueError("Cliente visitado por mais de um veículo")
        result.append(period)
    return result


def compact_solution(problem, variables):
    routes = routes_from_z(
        variables["Z"], shape=(problem.v, problem.i, problem.k, problem.t)
    )
    compact = {k: np.array(variables[k], copy=True) for k in ("X", "Y", "I", "Q")}
    compact["Y"] = np.rint(compact["Y"]).astype(int)
    compact.update(
        route_plan=routes,
        assignments=[
            {i: v for v, route in enumerate(period) for i in route} for period in routes
        ],
        feasible=True,
    )
    return compact


def materialize_snapshot(problem, variables):
    """Allocate dense arc tensors only for output/debug or a dense handoff."""
    if not isinstance(variables["Z"], dict):
        return variables
    result = dict(variables)
    for name, shape in (
        ("Z", (problem.v, problem.i, problem.k, problem.t)),
        ("R", (problem.p, problem.v, problem.i, problem.k, problem.t)),
    ):
        result[name] = np.zeros(shape, dtype=float)
        for key, value in variables[name].items():
            result[name][key] = value
    return result


def materialize_routes(problem, compact):
    result = empty_solution(problem)
    for name in ("X", "Y", "I", "Q"):
        result[name][...] = compact[name]
    result["eta"] = np.zeros((problem.v, problem.i, problem.t), dtype=int)
    for t, routes in enumerate(compact["route_plan"]):
        for v, route in enumerate(routes):
            if not route:
                continue
            result["eta"][v, [0, *route], t] = 1
            nodes = [0, *route, 0]
            load = result["Q"][:, v, route, t].sum(axis=1)
            for a, b in zip(nodes, nodes[1:]):
                result["Z"][v, a, b, t] = 1
                result["R"][:, v, a, b, t] = load
                if b:
                    load = load - result["Q"][:, v, b, t]
    return result


def canonicalize_vehicles(problem, variables):
    result = clone_solution(variables)
    routes = routes_from_z(
        variables["Z"], shape=(problem.v, problem.i, problem.k, problem.t)
    )
    orders = [
        sorted(
            range(problem.v), key=lambda v: (min(period[v]) if period[v] else np.inf, v)
        )
        for period in routes
    ]
    for name, axis in (("Q", 1), ("R", 1), ("Z", 0), ("eta", 0)):
        if name not in result:
            continue
        if isinstance(variables[name], dict):
            inverse = [{v: rank for rank, v in enumerate(order)} for order in orders]
            result[name] = {
                (*key[:axis], inverse[key[-1]][key[axis]], *key[axis + 1 :]): value
                for key, value in variables[name].items()
            }
        else:
            for t, order in enumerate(orders):
                result[name][..., t] = np.take(
                    np.asarray(variables[name])[..., t], order, axis=axis
                )
    return result


def validate_variables(problem, variables, allowed_arcs=None, tolerance=1e-6):
    """Audit quantities, connectivity and loads without dense arc allocation."""
    shapes = {
        "X": (problem.p, problem.t),
        "Y": (problem.p, problem.t),
        "I": (problem.p, problem.i, problem.t),
        "Q": (problem.p, problem.v, problem.i, problem.t),
    }
    violations = []
    for name, shape in shapes.items():
        array = np.asarray(variables[name])
        if array.shape != shape:
            return {"feasible": False, "violations": [f"{name}: dimensões inválidas"]}
        if not np.all(np.isfinite(array)) or np.any(array < -tolerance):
            violations.append(f"{name}: valor não finito ou negativo")
    try:
        routes = routes_from_z(
            variables["Z"],
            tolerance,
            shape=(problem.v, problem.i, problem.k, problem.t),
        )
    except ValueError as error:
        return {"feasible": False, "violations": violations + [str(error)]}
    x, y, stock, q = (np.asarray(variables[k]) for k in ("X", "Y", "I", "Q"))
    if np.any(y > 1 + tolerance) or not np.allclose(
        y, np.rint(y), atol=tolerance, rtol=0
    ):
        violations.append("Setup não binário")
    if np.any(x > problem.M * y + tolerance) or np.any(
        np.asarray(problem.b_p) @ x > problem.B + tolerance
    ):
        violations.append("Produção/setup/capacidade")
    if np.any(stock > np.asarray(problem.U_p_i)[:, :, None] + tolerance):
        violations.append("Capacidade de estoque")
    previous = np.concatenate(
        (np.asarray(problem.I_p_i_0)[:, :, None], stock[:, :, :-1]), axis=2
    )
    if not np.allclose(
        previous[:, 0] + x - q[:, :, 1:].sum(axis=(1, 2)),
        stock[:, 0],
        atol=tolerance,
        rtol=1e-8,
    ):
        violations.append("Balanço de estoque na planta")
    if not np.allclose(
        previous[:, 1:] + q[:, :, 1:].sum(axis=1) - problem.d_p_i_t,
        stock[:, 1:],
        atol=tolerance,
        rtol=1e-8,
    ):
        violations.append("Balanço de estoque nos clientes")
    if np.any(np.abs(q[:, :, 0]) > tolerance):
        violations.append("Entrega ao depósito")
    flow = np.zeros_like(q)
    loads = {}
    if not isinstance(variables["R"], dict) and np.asarray(variables["R"]).shape != (
        problem.p,
        problem.v,
        problem.i,
        problem.k,
        problem.t,
    ):
        return {
            "feasible": False,
            "violations": violations + ["R: dimensões inválidas"],
        }
    for (p, v, i, j, t), value in arc_items(variables["R"]):
        if not (
            0 <= p < problem.p
            and 0 <= v < problem.v
            and 0 <= i < problem.i
            and 0 <= j < problem.k
            and 0 <= t < problem.t
        ):
            return {
                "feasible": False,
                "violations": violations + ["R: índice inválido"],
            }
        if not np.isfinite(value) or value < -tolerance:
            violations.append("R: valor não finito ou negativo")
        flow[p, v, j, t] += value
        flow[p, v, i, t] -= value
        key = (v, i, j, t)
        loads[key] = loads.get(key, 0) + value
    if not np.allclose(flow[:, :, 1:], q[:, :, 1:], atol=tolerance, rtol=1e-8):
        violations.append("Balanço de fluxo nos clientes")
    z = variables["Z"]
    for key, load in loads.items():
        value = z.get(key, 0) if isinstance(z, dict) else z[key]
        if load > problem.C * value + tolerance:
            violations.append("Capacidade/ligação carga-arco")
            break
    expected_eta = np.zeros((problem.v, problem.i, problem.t))
    for t, period in enumerate(routes):
        for v, route in enumerate(period):
            if route:
                expected_eta[v, [0, *route], t] = 1
            unvisited = set(range(1, problem.i)) - set(route)
            if unvisited and np.any(q[:, v, sorted(unvisited), t] > tolerance):
                violations.append("Entrega sem visita")
    if "eta" in variables and (
        np.asarray(variables["eta"]).shape != expected_eta.shape
        or not np.allclose(variables["eta"], expected_eta, atol=tolerance, rtol=0)
    ):
        violations.append("Graus inconsistentes com eta")
    if allowed_arcs is not None:
        allowed = set(allowed_arcs)
        if any(
            value > tolerance and (i, j) not in allowed
            for (v, i, j, t), value in arc_items(z)
        ):
            violations.append("Arco fora da ordem")
    return {"feasible": not violations, "violations": violations}
