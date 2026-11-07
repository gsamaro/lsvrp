"""First-need setup inequalities (``solver.first_need_setup_cuts``).

A product with a positive requirement up to period ``l`` must have a setup at or
before ``l``. The inequality is valid for every integer solution and cuts only
fractional setups, so the optimum is unchanged and the linear relaxation rises.
"""

import importlib.util
import json
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from config import Config
from src.process.TargetGenerationProcess import effective_config, generation_settings
from src.solvers._solver_common import ProblemData, first_need_periods
from src.solvers.MultProductProdctionRoutingProblem import (
    MultProductProdctionRoutingProblem as MPPRP,
)

HAS_CPLEX = importlib.util.find_spec("cplex") is not None
needs_cplex = pytest.mark.skipif(not HAS_CPLEX, reason="CPLEX not installed")


def _data(demand, initial_customers=None, initial_plant=None, capacity=100, vehicle=50):
    """``demand[p][i][t]`` for customers ``i = 0..n-1``; two periods or more."""
    products, customers, periods = len(demand), len(demand[0]), len(demand[0][0])
    nodes = customers + 1
    coordinates = [(0, 0), (4, 0), (8, 0), (0, 4)][:nodes]
    initial = [
        [(initial_plant or [0] * products)[p]]
        + list((initial_customers or [[0] * customers] * products)[p])
        for p in range(products)
    ]
    return {
        "file": "unit.dat",
        "num_products": products,
        "num_customers": customers,
        "num_periods": periods,
        "num_vehicles": 2,
        "B": capacity,
        "b_p": [1] * products,
        "c_p": [1] * products,
        "s_p": [10] * products,
        "M": 1000,
        "U_pi": [[500] * nodes for _ in range(products)],
        "I_pi0": initial,
        "h_pi": [[0] + [1] * customers for _ in range(products)],
        "C": vehicle,
        "f": 20,
        "a_ik": [
            [abs(a[0] - b[0]) + abs(a[1] - b[1]) for b in coordinates]
            for a in coordinates
        ],
        "d_pit": demand,
        "weight": [0, 1, 0, 0, 0],
        "alpha": 0.5,
        "targets": [
            {f"f{j}_target": 10.0 * j for j in range(1, 6)} for _ in range(periods)
        ],
    }


def _configure(monkeypatch, cuts=True, multiobjective=False, target=None, **solver):
    settings = {
        "multiobjective": multiobjective,
        "method": "GUROBY",
        "aggregate_vehicles": True,
        "symmetric_routes": True,
        "first_need_setup_cuts": cuts,
        **solver,
    }
    value = {"solver": settings, "relaxed_solution": {"use": False}}
    if target is not None:
        value["postprocessing"] = {"build_target": True}
        value["relaxed_solution"] = {
            "use": False,
            "replace_model": True,
            "target_use_relaxation": target == "lp",
        }
    monkeypatch.setattr(Config, "_data", Config.normalize(value))


# ------------------------------------------------------------------ first need


def test_first_need_is_the_first_period_with_a_positive_requirement():
    #                  customer 0      customer 1     customer 2
    demand = [[[0, 0, 2, 2], [0, 1, 0, 0], [0, 0, 0, 0]], [[0] * 4] * 3]
    problem = ProblemData.from_map(_data(demand))
    assert first_need_periods(problem).tolist() == [1, -1]  # product 1 never needed


def test_initial_stock_of_customers_and_plant_postpones_the_need():
    demand = [[[0, 0, 2, 2], [0, 1, 0, 0], [0, 0, 0, 0]]]
    customers_stock = [[0, 1, 0]]  # customer 1 covers its own period-1 demand
    assert first_need_periods(
        ProblemData.from_map(_data(demand, initial_customers=customers_stock))
    ).tolist() == [2]
    # The plant stock covers part of what is left: 1 of the 2 units needed by period 2.
    assert first_need_periods(
        ProblemData.from_map(_data(demand, initial_plant=[1]))
    ).tolist() == [2]
    assert first_need_periods(
        ProblemData.from_map(_data(demand, initial_plant=[10]))
    ).tolist() == [-1]


def test_stock_of_one_customer_cannot_serve_another():
    demand = [[[3, 0], [0, 0]]]
    stock = [[0, 50]]  # all the stock is at the customer that needs nothing
    assert first_need_periods(
        ProblemData.from_map(_data(demand, initial_customers=stock))
    ).tolist() == [0]


# ------------------------------------------------------------------------ model


def test_one_inequality_per_product_up_to_its_first_need(monkeypatch):
    demand = [[[0, 0, 2, 2], [0, 1, 0, 0], [0, 0, 0, 0]], [[0] * 4] * 3]
    _configure(monkeypatch)
    solver = MPPRP(_data(demand), "/tmp", Mock(), {"start": False})
    solver.createDecisionVariables()
    assert solver.createFirstNeedSetups() is True
    names = [c.name for c in solver.model.iter_constraints() if c.name]
    assert names == ["FIRST_NEED_SETUP_p_0_l_1"]  # product 1 has no demand
    constraint = solver.model.get_constraint_by_name("FIRST_NEED_SETUP_p_0_l_1")
    assert constraint.rhs.constant == 1
    assert {v.name for v in constraint.lhs.iter_variables()} == {"Y_0_0", "Y_0_1"}
    solver.terminate()


def test_the_inequalities_follow_the_flag_and_skip_a_fixed_plan(monkeypatch):
    demand = [[[3, 1], [2, 2]]]
    _configure(monkeypatch, cuts=False)
    off = MPPRP(_data(demand), "/tmp", Mock(), {"start": False})
    off.createDecisionVariables()
    assert off.createFirstNeedSetups() is False
    off.terminate()

    _configure(monkeypatch, cuts=False)
    forced = MPPRP({**_data(demand), "first_need_setup_cuts": True}, "/tmp", Mock(), {"start": False})
    forced.createDecisionVariables()
    assert forced.createFirstNeedSetups() is True  # the data map takes precedence
    forced.terminate()

    _configure(monkeypatch)
    plan = {"X": np.zeros((1, 2)), "Y": np.zeros((1, 2)), "delivery_totals": np.zeros((1, 3, 2))}
    fixed = MPPRP(_data(demand), "/tmp", Mock(), {"start": False}, fixed_plan=plan)
    fixed.createDecisionVariables()
    assert fixed.createFirstNeedSetups() is False  # the setups are already fixed
    fixed.terminate()


@needs_cplex
def test_applied_features_record_the_option(monkeypatch):
    _configure(monkeypatch)
    solver = MPPRP(_data([[[3, 1], [2, 2]]]), "/tmp", Mock(), {"start": False})
    solver.solver(timeLimit=5, numThreads=1)
    assert solver.applied_features["first_need_setup_cuts"] is True
    solver.terminate()


# ----------------------------------------------------------------------- config


def test_default_is_on_and_validated():
    assert Config.normalize({})["solver"]["first_need_setup_cuts"] is True
    assert Config.normalize({})["target_generation"]["formulation"]["first_need_setup_cuts"] is True
    with pytest.raises(ValueError, match="first_need_setup_cuts"):
        Config.normalize({"solver": {"first_need_setup_cuts": "sim"}})


def test_target_generation_formulation_is_validated_and_forwarded():
    configured = Config.normalize({})
    configured["target_generation"]["formulation"]["first_need_setup_cuts"] = "sim"
    with pytest.raises(ValueError, match="first_need_setup_cuts"):
        Config.normalize(configured)

    for value in (True, False):
        configured = Config.normalize({})
        configured["target_generation"]["formulation"]["first_need_setup_cuts"] = value
        monkey = Config.normalize(configured)
        resolved = effective_config(monkey["target_generation"] | {"instance": {"files": ["DATA_PRP_5C"], "dir": "./data/"}})
        assert resolved["solver"]["first_need_setup_cuts"] is value


def test_shipped_config_declares_the_option():
    path = Path(__file__).resolve().parents[1] / "config" / "config.json"
    shipped = json.loads(path.read_text(encoding="utf-8"))
    assert shipped["solver"]["first_need_setup_cuts"] is True
    assert shipped["target_generation"]["formulation"]["first_need_setup_cuts"] is True


# ------------------------------------------------------- validity and strength


# Two products of 10 units each share a production capacity of 15 per period: the
# linear relaxation pays half a setup for the 5 units of the second product at
# period 0, while any integer plan pays a whole one.
SHARED_CAPACITY = [
    [[3, 2], [2, 3]],
    [[3, 2], [2, 3]],
]


def _f2_only(monkeypatch, cuts, lp):
    _configure(monkeypatch, cuts=cuts, target="lp" if lp else "mip")
    data = _data(SHARED_CAPACITY, capacity=15)
    solver = MPPRP(data, "/tmp", Mock(), {"start": False})
    solver.solver(timeLimit=30, numThreads=1)
    value = solver.relaxedModelObjVal if lp else solver.model.objective_value
    solver.terminate()
    return value


@needs_cplex
def test_the_relaxation_pays_a_whole_setup_and_the_optimum_is_unchanged(monkeypatch):
    lp_without = _f2_only(monkeypatch, cuts=False, lp=True)
    lp_with = _f2_only(monkeypatch, cuts=True, lp=True)
    mip_without = _f2_only(monkeypatch, cuts=False, lp=False)
    mip_with = _f2_only(monkeypatch, cuts=True, lp=False)

    assert mip_with == pytest.approx(mip_without)  # valid: no integer plan is cut off
    assert mip_with == pytest.approx(30.0)  # three setups of 10
    assert lp_with > lp_without + 1e-6  # and the relaxation rises
    assert lp_with <= mip_with + 1e-6  # staying a valid bound


@needs_cplex
@pytest.mark.parametrize("multiobjective", [False, True])
def test_the_optimum_of_the_full_model_is_unchanged(monkeypatch, multiobjective):
    demand = [
        [[3, 1, 2], [2, 2, 1], [1, 3, 2]],
        [[2, 2, 1], [1, 1, 3], [2, 0, 2]],
    ]
    data = _data(demand, initial_customers=[[0, 1, 0], [1, 0, 0]], capacity=14, vehicle=8)
    data["weight"] = [0.2] * 5
    values = []
    for cuts in (False, True):
        _configure(monkeypatch, cuts=cuts, multiobjective=multiobjective)
        solver = MPPRP(dict(data), "/tmp", Mock(), {"start": False})
        solver.solver(timeLimit=30, numThreads=1)
        values.append(solver.model.objective_value)
        solver.terminate()
    assert values[1] == pytest.approx(values[0], rel=1e-6)
