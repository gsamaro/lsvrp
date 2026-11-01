"""Ideal, nadir e targets de uma instância a partir da relaxação linear do modelo.

O modelo é o mesmo da construção de targets (``build_target`` com relaxação): o
``MultProductProdctionRoutingProblem`` é montado e relaxado pelo próprio
``solver()`` e as cinco execuções monoobjetivo ficam sobre esse mesmo LP.

* ideal por período: ``ideal[j, t] = min_k f_{j,t}(vértice de k)``, o vértice que o
  simplex devolve ao minimizar ``sum_t f_k``;
* nadir ``"runs_max"``: o máximo dos mesmos vértices (o ``max`` de ``build_target``);
* nadir ``"face"``: ``nadir[j, t] = max(ideal[j, t], max_{k != j} min_{x em F_k} f_{j,t}(x))``,
  com a face ótima ``F_k = {x : sum_t f_k <= z_k (1 + eps) + eps}``. O mínimo de um
  LP tem valor único, então esse termo não depende do vértice devolvido;
* target: ``ideal + factor * (nadir - ideal)``.

O ideal depende do vértice (ótimos alternativos), por isso o algoritmo do LP é
fixado em ``lp.algorithm`` e a versão do CPLEX é registrada junto dos resultados.
"""

import numpy as np
import pandas as pd

OBJECTIVES = 5
TARGET_FACTOR = 0.3
FACE_EPSILON = 1e-6
NADIR_FACE = "face"
NADIR_RUNS_MAX = "runs_max"
NADIR_METHODS = (NADIR_FACE, NADIR_RUNS_MAX)
LP_ALGORITHMS = {"dual": 2, "primal": 1}
TARGET_COLUMNS = (
    [f"f{j}_ideal" for j in range(1, OBJECTIVES + 1)]
    + [f"f{j}_nadir" for j in range(1, OBJECTIVES + 1)]
    + [f"f{j}_target" for j in range(1, OBJECTIVES + 1)]
)
BASE_COLUMNS = ["file", "time"]


# Retries for a face LP the configured algorithm leaves with a numerical status (the
# face is a row with a slack of ~1e-6 over costs of ~1e5). The minimum of an LP is
# unique, so any algorithm that ends "optimal" returns the same value.
RETRY_STEPS = (
    ("dual+numerical", {"lpmethod": 2, "numerical": 1, "feasibility": 1e-9}),
    ("primal+numerical", {"lpmethod": 1, "numerical": 1, "feasibility": 1e-9}),
    ("barrier+numerical", {"lpmethod": 4, "numerical": 1, "feasibility": 1e-9}),
)


class _NullLog:
    def __getattr__(self, name):
        return lambda *args, **kwargs: None


def ideal_from_vertices(vertex):
    """Ideal ``(J, T)`` a partir dos vértices ``(K, J, T)`` das cinco execuções."""
    vertex = np.asarray(vertex, dtype=float)
    _check_cube(vertex)
    return vertex.min(axis=0)


def nadir_from_vertices(vertex):
    """Nadir ``(J, T)`` pelo máximo dos vértices (``build_target`` antigo)."""
    vertex = np.asarray(vertex, dtype=float)
    _check_cube(vertex)
    return vertex.max(axis=0)


def nadir_from_face_minima(face_minima, ideal):
    """Nadir ``(J, T)`` a partir dos mínimos nas faces ``(K, J, T)`` e do ideal.

    ``face_minima[k, j, t]`` é o menor ``f_{j,t}`` na face ótima de ``k``; a
    diagonal ``k == j`` é ignorada (NaN). O nadir nunca fica abaixo do ideal.
    """
    face_minima = np.asarray(face_minima, dtype=float)
    ideal = np.asarray(ideal, dtype=float)
    if face_minima.shape != (OBJECTIVES, *ideal.shape) or ideal.shape[0] != OBJECTIVES:
        raise ValueError("face_minima deve ter forma (5, 5, T) e ideal (5, T)")
    masked = face_minima.copy()
    for j in range(OBJECTIVES):
        masked[j, j, :] = np.nan
    if np.isnan(masked).sum(axis=0).max() > 1:
        raise ValueError("face_minima contém NaN fora da diagonal")
    return np.maximum(np.nanmax(masked, axis=0), ideal)


def target_from_ideal_nadir(ideal, nadir, factor=TARGET_FACTOR):
    return np.asarray(ideal, dtype=float) + factor * (
        np.asarray(nadir, dtype=float) - np.asarray(ideal, dtype=float)
    )


def _check_cube(vertex):
    if vertex.ndim != 3 or vertex.shape[:2] != (OBJECTIVES, OBJECTIVES):
        raise ValueError("vertex deve ter forma (5, 5, T)")


def instance_targets(vertex, face_minima, nadir_method, factor=TARGET_FACTOR):
    """``(ideal, nadir, target)``, cada um ``(J, T)``, de uma instância."""
    ideal = ideal_from_vertices(vertex)
    if nadir_method == NADIR_FACE:
        if face_minima is None:
            raise ValueError("nadir 'face' exige os mínimos nas faces")
        nadir = nadir_from_face_minima(face_minima, ideal)
    elif nadir_method == NADIR_RUNS_MAX:
        nadir = nadir_from_vertices(vertex)
    else:
        raise ValueError(f"método de nadir desconhecido: {nadir_method}")
    return ideal, nadir, target_from_ideal_nadir(ideal, nadir, factor)


def target_rows(file_key, ideal, nadir, target):
    """Linhas de ``targets.xlsx`` (uma por período) com as colunas históricas."""
    periods = ideal.shape[1]
    frame = {"file": [file_key] * periods, "time": list(range(periods))}
    for suffix, values in (("ideal", ideal), ("nadir", nadir), ("target", target)):
        for j in range(OBJECTIVES):
            frame[f"f{j + 1}_{suffix}"] = values[j]
    return pd.DataFrame(frame)[BASE_COLUMNS + TARGET_COLUMNS]


def validate_target_rows(rows, periods):
    """Falha se a instância tem NaN, período faltando ou nadir abaixo do ideal."""
    if len(rows) != periods or rows["time"].tolist() != list(range(periods)):
        raise ValueError(f"{rows['file'].iloc[0]}: períodos incompletos")
    values = rows[TARGET_COLUMNS]
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError(f"{rows['file'].iloc[0]}: targets não finitos")
    for j in range(1, OBJECTIVES + 1):
        if (rows[f"f{j}_nadir"] < rows[f"f{j}_ideal"] - 1e-9).any():
            raise ValueError(f"{rows['file'].iloc[0]}: f{j} com nadir abaixo do ideal")


def library_versions():
    versions = {}
    for name in ("cplex", "docplex"):
        try:
            module = __import__(name)
            versions[name] = str(getattr(module, "__version__", "desconhecida"))
        except ImportError:
            versions[name] = None
    return versions


def period_expressions(problem):
    """``f1..f5`` por período sobre as variáveis do modelo relaxado do ``problem``.

    Espelha ``crateObjectiveFunction``: depois de ``generteRelax(REPLACE_MODEL=True)``
    as expressões guardadas em ``problem.f*`` apontam para o modelo descartado.
    """
    model = problem.model
    periods = problem.t
    f = [[None] * periods for _ in range(OBJECTIVES)]
    for t in range(periods):
        f[0][t] = model.sum(problem.c_p[p] * model.X_p_t[p, t] for p in range(problem.p))
        f[1][t] = model.sum(problem.s_p[p] * model.Y_p_t[p, t] for p in range(problem.p))
        f[2][t] = model.sum(
            problem.h_p_i[p][i] * model.I_p_i_t[p, i, t]
            for p in range(problem.p)
            for i in range(problem.i)
        )
        if problem.symmetric_routes:
            f[3][t] = problem.f * problem._route_count(t)
            f[4][t] = model.sum(
                problem.a_i_k[i][k] * model.Xe_i_k_t[i, k, t] for i, k in problem.edges
            )
        else:
            f[3][t] = model.sum(
                problem.f * model.Z_v_i_k_t[v, 0, k, t]
                for v in range(problem.v_model)
                for k in problem.outgoing[t][0]
            )
            f[4][t] = model.sum(
                problem.a_i_k[i][k] * model.Z_v_i_k_t[v, i, k, t]
                for v in range(problem.v_model)
                for i, k in problem.arcs_by_period[t]
            )
    return f


def _lp_status(model):
    return str(getattr(model.solve_details, "status", "")).lower()


def _is_clean_optimal(solution, status):
    return solution is not None and status.startswith("optimal") and (
        "infeasib" not in status
    )


def solve_lp(model, expression, label, base_lpmethod, retries=None, steps=RETRY_STEPS):
    """Minimize ``expression``; with ``retries`` (a list) try ``steps`` if not optimal.

    Without ``retries`` a status other than a clean ``optimal`` raises. With it, each
    step changes the LP algorithm and numerical settings, and the settings of the run
    (``base_lpmethod``, default numerics) are restored afterwards; a step that worked
    is appended to ``retries``.
    """
    model.minimize(expression)
    solution = model.solve()
    status = _lp_status(model)
    if _is_clean_optimal(solution, status):
        return float(solution.objective_value)
    trail = [status]
    if retries is not None:
        parameters = model.context.cplex_parameters
        try:
            for name, settings in steps:
                parameters.lpmethod = settings["lpmethod"]
                parameters.emphasis.numerical = settings["numerical"]
                parameters.simplex.tolerances.feasibility = settings["feasibility"]
                solution = model.solve()
                status = _lp_status(model)
                trail.append(f"{name}: {status}")
                if _is_clean_optimal(solution, status):
                    retries.append({"lp": label, "step": name, "first": trail[0]})
                    return float(solution.objective_value)
        finally:
            parameters.lpmethod = base_lpmethod
            parameters.emphasis.numerical = 0
            parameters.simplex.tolerances.feasibility = 1e-6
    raise RuntimeError(f"LP sem solução ótima ({label}): {'; '.join(trail)}")


def compute_instance_lps(
    instance,
    weights,
    nadir_method=NADIR_FACE,
    epsilon=FACE_EPSILON,
    lp_algorithm="dual",
    alpha=0.99,
):
    """Resolve os LPs de uma instância com a ``Config`` já instalada. Exige CPLEX.

    ``weights`` são as cinco linhas unitárias; a linha ``r`` minimiza o objetivo
    ``argmax(weights[r])``. Devolve um dicionário com ``z[k]`` (ótimo isolado de
    ``f_k``, soma no horizonte), ``vertex[k, j, t]`` (``f_{j,t}`` no vértice de
    ``k``), ``face_minima[k, j, t]`` (NaN em ``k == j``; ``None`` sem o método
    ``face``) e dados de diagnóstico.
    """
    import time

    from src.helpers.ReadPrpFile import ReadPrpFile
    from src.solvers.MultProductProdctionRoutingProblem import (
        MultProductProdctionRoutingProblem,
    )
    from config import Config

    started = time.time()
    log = _NullLog()
    objective_of_row = [int(np.argmax(row)) for row in weights]
    data = ReadPrpFile(file_path=instance, log=log).getDataSet()
    data["weight"] = list(weights[0])
    data["alpha"] = alpha
    data["strengthened_bounds"] = Config.get_nested(
        "solver", "strengthened_bounds", default=True
    )
    data["targets"] = []
    problem = MultProductProdctionRoutingProblem(
        map=data, dir="", log=log, start={"start": False}
    )
    problem.solver(numThreads=1)
    if not problem._relaxation_replaced:
        raise RuntimeError(f"{instance}: relaxação não substituiu o modelo")
    model = problem.model
    periods = problem.t
    first_solve_value = float(problem.relaxedModelObjVal)
    build_seconds = time.time() - started

    model.context.cplex_parameters.threads = 1
    model.context.cplex_parameters.lpmethod = LP_ALGORITHMS[lp_algorithm]
    f = period_expressions(problem)

    retries = []
    base_lpmethod = LP_ALGORITHMS[lp_algorithm]

    def minimize(expression, label, face=False):
        try:
            return solve_lp(
                model, expression, label, base_lpmethod, retries if face else None
            )
        except RuntimeError as error:
            raise RuntimeError(f"{instance}: {error}") from error

    z = np.zeros(OBJECTIVES)
    vertex = np.zeros((OBJECTIVES, OBJECTIVES, periods))
    for row in range(OBJECTIVES):
        k = objective_of_row[row]
        z[k] = minimize(model.sum(f[k]), f"base f{k + 1}")
        for j in range(OBJECTIVES):
            for t in range(periods):
                vertex[k, j, t] = f[j][t].solution_value
    reference = z[objective_of_row[0]]
    if abs(reference - first_solve_value) > 1e-6 * max(1.0, abs(reference)):
        raise RuntimeError(
            f"{instance}: LP do solver ({first_solve_value}) difere do LP das "
            f"expressões f ({reference})"
        )

    face_minima = None
    if nadir_method == NADIR_FACE:
        face_minima = np.full((OBJECTIVES, OBJECTIVES, periods), np.nan)
        for k in range(OBJECTIVES):
            face = model.add_constraint(
                model.sum(f[k]) <= z[k] * (1 + epsilon) + epsilon, ctname=f"face_{k}"
            )
            for j in range(OBJECTIVES):
                if j == k:
                    continue
                for t in range(periods):
                    face_minima[k, j, t] = minimize(
                        f[j][t], f"face f{k + 1} -> f{j + 1} t{t}", face=True
                    )
            model.remove_constraint(face)

    features = dict(getattr(problem, "applied_features", {}))
    model.end()
    return {
        "z": z,
        "vertex": vertex,
        "face_minima": face_minima,
        "periods": periods,
        "retried_lps": retries,
        "features": features,
        "seconds": time.time() - started,
        "build_seconds": build_seconds,
    }
