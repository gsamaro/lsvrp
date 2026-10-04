"""Explicit unit-only runner: no CPLEX model can be solved in this mode."""

import numpy as np
import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--unit-only",
        action="store_true",
        help="Deseleciona integração CPLEX e bloqueia Model.solve",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "cplex_integration: resolve um modelo CPLEX real"
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--unit-only"):
        deselected = [
            item for item in items if item.get_closest_marker("cplex_integration")
        ]
        items[:] = [
            item for item in items if not item.get_closest_marker("cplex_integration")
        ]
        config.hook.pytest_deselected(items=deselected)


@pytest.fixture(autouse=True)
def forbid_cplex_solves(request, monkeypatch):
    if request.config.getoption("--unit-only"):
        from docplex.mp.model import Model

        def forbidden(*args, **kwargs):
            raise AssertionError("Resolução CPLEX proibida em --unit-only; use um mock")

        monkeypatch.setattr(Model, "solve", forbidden)


@pytest.fixture(autouse=True)
def pso_lot_sizing_double(request, monkeypatch):
    # These PSO tests exercise swarm behavior, not optimization of the LP.
    if request.config.getoption("--unit-only") and request.module.__name__.endswith(
        "test_pso_solver"
    ):
        from src.solvers.LotSizingRelaxation import LotSizingRelaxation

        def bounds(self, include_base=False):
            p = self.problem
            q = np.zeros((p.p, p.v, p.i, p.t))
            q[:, 0, 1:, :] = np.asarray(p.d_p_i_t)
            x = q.sum(axis=(1, 2))
            lower = {"X": x, "Q": q}
            upper = {
                "X": np.maximum(x, np.minimum(p.B, np.asarray(p.U_p_i)[:, 0, None])),
                "Q": q.copy(),
            }
            return {
                "lower": float(x.sum()),
                "upper": float(upper["X"].sum()),
                "lower_gap": 0,
                "upper_gap": 0,
                "lower_solution": lower,
                "upper_solution": upper,
                "base": lower if include_base else None,
            }

        monkeypatch.setattr(LotSizingRelaxation, "solve_bounds", bounds)
