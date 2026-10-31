import importlib.util

import numpy as np
import pandas as pd
import pytest

from config import Config
from src.helpers.FaceNadirTargets import (
    TARGET_COLUMNS,
    compute_instance_lps,
    ideal_from_vertices,
    instance_targets,
    nadir_from_face_minima,
    nadir_from_vertices,
    target_from_ideal_nadir,
    target_rows,
    validate_target_rows,
)

WEIGHTS = [[1 if j == k else 0 for j in range(5)] for k in range(5)]


def _minima(periods, value):
    minima = np.full((5, 5, periods), value, dtype=float)
    for j in range(5):
        minima[j, j, :] = np.nan
    return minima


def _vertex(periods):
    # vertex[k, j, t] = 100 * k + 10 * j + t
    return np.fromfunction(lambda k, j, t: 100 * k + 10 * j + t, (5, 5, periods))


def test_ideal_is_the_per_period_minimum_over_the_five_runs():
    vertex = _vertex(2)
    assert np.array_equal(ideal_from_vertices(vertex), vertex[0])


def test_runs_max_nadir_is_the_per_period_maximum_over_the_five_runs():
    vertex = _vertex(2)
    assert np.array_equal(nadir_from_vertices(vertex), vertex[4])


def test_vertices_need_the_five_by_five_by_periods_shape():
    with pytest.raises(ValueError):
        ideal_from_vertices(np.zeros((4, 5, 2)))


def test_nadir_is_worst_unavoidable_damage_over_other_objectives():
    minima = _minima(2, 10.0)
    minima[2, 0, 1] = 40.0  # face de f3 obriga f1 no período 1 a ser >= 40
    minima[3, 0, 1] = 25.0
    ideal = np.zeros((5, 2))

    nadir = nadir_from_face_minima(minima, ideal)

    assert nadir[0, 0] == 10.0
    assert nadir[0, 1] == 40.0


def test_nadir_ignores_own_face_and_never_falls_below_ideal():
    minima = _minima(1, 1.0)
    minima[0, 0, 0] = 1000.0  # diagonal: deve ser ignorada
    ideal = np.full((5, 1), 5.0)

    nadir = nadir_from_face_minima(minima, ideal)

    assert np.all(nadir == 5.0)


def test_nadir_rejects_missing_minima_outside_the_diagonal():
    minima = _minima(1, 1.0)
    minima[1, 0, 0] = np.nan
    with pytest.raises(ValueError):
        nadir_from_face_minima(minima, np.zeros((5, 1)))


def test_target_is_ideal_plus_factor_of_range():
    target = target_from_ideal_nadir([10.0, 4.0], [20.0, 4.0], 0.3)
    assert target.tolist() == [13.0, 4.0]


def test_instance_targets_follow_the_chosen_nadir_method():
    vertex = _vertex(2)
    face = _minima(2, 1000.0)

    ideal, nadir, target = instance_targets(vertex, face, "face", 0.5)
    assert np.all(nadir == 1000.0)
    assert np.allclose(target, ideal + 0.5 * (1000.0 - ideal))

    _, nadir_runs, _ = instance_targets(vertex, None, "runs_max", 0.5)
    assert np.array_equal(nadir_runs, vertex[4])

    with pytest.raises(ValueError):
        instance_targets(vertex, None, "face")
    with pytest.raises(ValueError):
        instance_targets(vertex, None, "outro")


def test_target_rows_keep_the_historical_columns_and_order():
    ideal = np.arange(10, dtype=float).reshape(5, 2)
    rows = target_rows("./data/DATA_PRP_5C/a.dat", ideal, ideal + 1, ideal + 0.3)

    assert list(rows.columns) == ["file", "time"] + TARGET_COLUMNS
    assert rows["time"].tolist() == [0, 1]
    assert rows["f2_ideal"].tolist() == [2.0, 3.0]
    assert rows["f5_nadir"].tolist() == [9.0, 10.0]


def test_validate_rejects_nan_missing_period_and_inverted_range():
    ideal = np.ones((5, 2))
    good = target_rows("a.dat", ideal, ideal + 1, ideal + 0.3)
    validate_target_rows(good, 2)

    with pytest.raises(ValueError):
        validate_target_rows(good.iloc[:1], 2)
    broken = good.copy()
    broken.loc[0, "f3_target"] = np.nan
    with pytest.raises(ValueError):
        validate_target_rows(broken, 2)
    inverted = good.copy()
    inverted.loc[1, "f1_nadir"] = 0.0
    with pytest.raises(ValueError):
        validate_target_rows(inverted, 2)


# --- LP real (CPLEX) -------------------------------------------------------------

INSTANCE = "./data/DATA_PRP_5C/PRP10_C5_P3_V1_T6_S1.dat"
cplex = pytest.mark.skipif(
    importlib.util.find_spec("cplex") is None,
    reason="CPLEX não está instalado neste ambiente",
)


def _generation_config(symmetric=True):
    from src.process.TargetGenerationProcess import effective_config, generation_settings

    settings = generation_settings(Config)
    settings["formulation"]["symmetric_routes"] = symmetric
    settings["formulation"]["aggregate_vehicles"] = True
    return effective_config(settings)


@pytest.fixture
def restore_config():
    saved = Config.snapshot()
    yield
    Config.replace(saved)


@cplex
@pytest.mark.cplex_integration
def test_lp_results_are_consistent_and_repeatable(restore_config):
    Config.replace(_generation_config())

    first = compute_instance_lps(INSTANCE, WEIGHTS)
    second = compute_instance_lps(INSTANCE, WEIGHTS)

    z, vertex, minima = first["z"], first["vertex"], first["face_minima"]
    assert vertex.shape == (5, 5, 6) and minima.shape == (5, 5, 6)
    assert (~np.isnan(minima)).sum() == 5 * 4 * 6
    # o vértice de f_k tem soma de f_k igual ao ótimo isolado
    for k in range(5):
        assert vertex[k, k, :].sum() == pytest.approx(z[k], rel=1e-6, abs=1e-6)
    # o vértice de k pertence à face de k: o mínimo na face não passa dele
    off_diagonal = ~np.isnan(minima)
    assert np.all(minima[off_diagonal] <= vertex[off_diagonal] + 1e-4)
    # repetir no mesmo ambiente devolve exatamente o mesmo ideal e o mesmo nadir
    assert np.array_equal(first["vertex"], second["vertex"])
    assert np.array_equal(
        first["face_minima"], second["face_minima"], equal_nan=True
    )


@cplex
@pytest.mark.cplex_integration
def test_face_minima_do_not_depend_on_the_lp_algorithm(restore_config):
    Config.replace(_generation_config())

    dual = compute_instance_lps(INSTANCE, WEIGHTS, lp_algorithm="dual")
    primal = compute_instance_lps(INSTANCE, WEIGHTS, lp_algorithm="primal")

    assert np.allclose(dual["z"], primal["z"], rtol=1e-6, atol=1e-6)
    assert np.allclose(
        dual["face_minima"], primal["face_minima"], rtol=1e-5, atol=1e-4, equal_nan=True
    )


@cplex
@pytest.mark.cplex_integration
def test_runs_max_nadir_skips_the_face_lps(restore_config):
    Config.replace(_generation_config())

    result = compute_instance_lps(INSTANCE, WEIGHTS, nadir_method="runs_max")

    assert result["face_minima"] is None
    assert result["vertex"].shape == (5, 5, 6)


@cplex
@pytest.mark.cplex_integration
def test_directed_and_symmetric_models_both_build_targets(restore_config):
    Config.replace(_generation_config(symmetric=True))
    symmetric = compute_instance_lps(INSTANCE, WEIGHTS, nadir_method="runs_max")
    Config.replace(_generation_config(symmetric=False))
    directed = compute_instance_lps(INSTANCE, WEIGHTS, nadir_method="runs_max")

    # produção, setup e estoque não dependem da formulação de rotas no LP ótimo isolado
    assert symmetric["features"]["symmetric_routes"] is True
    assert directed["features"]["symmetric_routes"] is False
    for k in range(3):
        assert symmetric["z"][k] == pytest.approx(directed["z"][k], rel=1e-6, abs=1e-6)
