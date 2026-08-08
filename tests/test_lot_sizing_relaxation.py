import numpy as np
import pytest

from src.solvers.FeasibleParticleHeuristic import FeasibleParticleHeuristic
from src.solvers.LotSizingRelaxation import LotSizingRelaxation


class DummyLogger:
    def info(self, message):
        pass

    def warning(self, message):
        pass

    def error(self, message):
        pass


def minimal_data(B=10):
    return {
        "num_products": 1,
        "num_customers": 1,
        "num_periods": 2,
        "num_vehicles": 1,
        "B": B,
        "b_p": [1],
        "c_p": [2],
        "s_p": [3],
        "M": 100,
        "U_pi": [[20, 20]],
        "I_pi0": [[0, 0]],
        "h_pi": [[0, 1]],
        "C": 10,
        "f": 7,
        "a_ik": [[0, 11], [11, 0]],
        "d_pit": [[[2, 3]]],
        "weight": [0.2] * 5,
        "alpha": 0.01,
    }


def test_relaxation_keeps_only_lot_sizing_constraints_and_calculates_bounds():
    relaxation = LotSizingRelaxation(minimal_data(), DummyLogger())
    model, _, _, _ = relaxation._build_model()
    names = [constraint.name for constraint in model.iter_constraints()]
    model.end()

    assert all("relax_" in name for name in names)
    assert not any("route" in name for name in names)
    assert "relax_vehicle_delivery_capacity_0_0" in names
    assert "relax_vehicle_delivery_capacity_0_1" in names

    result = relaxation.solve_bounds(include_base=True)

    assert result["lower"] == pytest.approx(5.0)
    assert result["upper"] == pytest.approx(20.0)
    assert result["lower"] <= result["upper"]
    assert np.sum(result["lower_solution"]["Q"]) == pytest.approx(5.0)
    np.testing.assert_allclose(
        result["base"]["X"],
        result["lower_solution"]["X"],
    )
    assert np.all(
        result["upper_solution"]["X"]
        >= result["lower_solution"]["X"] - 1e-8
    )
    assert np.all(
        result["upper_solution"]["Q"]
        >= result["lower_solution"]["Q"] - 1e-8
    )


def test_upper_model_contains_lower_production_constraints():
    relaxation = LotSizingRelaxation(minimal_data(), DummyLogger())
    lower = np.array([[2.0, 3.0]])
    lower_q = np.zeros((1, 1, 2, 2))
    lower_q[0, 0, 1, :] = [2.0, 3.0]
    model, _, _, _ = relaxation._build_model(
        production_lower_bounds=lower,
        quantity_lower_bounds=lower_q,
    )
    names = [constraint.name for constraint in model.iter_constraints()]
    model.end()

    assert "relax_production_lower_bound_0_0" in names
    assert "relax_production_lower_bound_0_1" in names
    assert "relax_delivery_lower_bound_0_0_1_0" in names
    assert "relax_delivery_lower_bound_0_0_1_1" in names


def test_relaxation_limits_total_delivery_per_vehicle_and_period():
    data = minimal_data()
    data["C"] = 4

    result = LotSizingRelaxation(data, DummyLogger()).solve_bounds()

    assert result["upper"] == pytest.approx(8.0)
    for t in range(data["num_periods"]):
        assert np.sum(result["upper_solution"]["Q"][0, 0, 1:, t]) <= 4.0 + 1e-8


def test_relaxation_reports_infeasible_instance():
    with pytest.raises(RuntimeError, match="PL relaxado sem solução"):
        LotSizingRelaxation(minimal_data(B=1), DummyLogger()).solve_bounds()


def test_relaxation_applies_configured_time_limit():
    relaxation = LotSizingRelaxation(minimal_data(), DummyLogger(), time_limit=7)
    model, _, _, _ = relaxation._build_model()
    relaxation._apply_time_limit(model)

    assert model.parameters.timelimit.get() == 7
    model.end()


def test_heuristic_decodes_production_inside_relaxed_bounds():
    data = minimal_data()
    bounds = {"lower": np.array([[2.0, 1.0]]), "upper": np.array([[6.0, 5.0]])}
    heuristic = FeasibleParticleHeuristic(data, "/tmp", DummyLogger(), bounds=bounds)

    assert heuristic._bounded_production_from_gene(0, 0, -100.0) == 2
    assert heuristic._bounded_production_from_gene(0, 0, 100.0) == 6

    high_gene = np.full((1, heuristic.particle_dim), 100.0)
    solution = heuristic.build_population(high_gene)[0]
    assert 2 <= solution["X"][0, 0] <= 6
    assert 1 <= solution["X"][0, 1] <= 5
    assert solution["feasible"]

    equal_bounds = {"lower": np.array([[3.0, 3.0]]), "upper": np.array([[3.0, 3.0]])}
    fixed = FeasibleParticleHeuristic(data, "/tmp", DummyLogger(), bounds=equal_bounds)
    assert fixed._bounded_production_from_gene(0, 1, 0.0) == 3
