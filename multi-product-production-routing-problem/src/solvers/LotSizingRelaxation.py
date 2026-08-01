"""Linear relaxation used to derive production bounds for the PSO."""

import math

import numpy as np
from docplex.mp.model import Model

from src.log.Logger import Logger
from src.solvers._solver_common import ProblemData


class LotSizingRelaxation:
    """Solve the production/inventory relaxation from constraints 8, 9, 10 and 12."""

    def __init__(self, map, log: Logger, time_limit=None):
        self.problem = ProblemData.from_map(map)
        self.log = log
        self.time_limit = time_limit

    def _build_model(self):
        problem = self.problem
        model = Model(name="PSO_LotSizing_Relaxation")
        x = {
            (p, t): model.continuous_var(lb=0, name=f"relax_X_{p}_{t}")
            for p in range(problem.p)
            for t in range(problem.t)
        }
        inventory = {
            (p, i, t): model.continuous_var(lb=0, name=f"relax_I_{p}_{i}_{t}")
            for p in range(problem.p)
            for i in range(problem.i)
            for t in range(problem.t)
        }
        quantity = {
            (p, v, i, t): model.continuous_var(lb=0, name=f"relax_Q_{p}_{v}_{i}_{t}")
            for p in range(problem.p)
            for v in range(problem.v)
            for i in range(problem.i)
            for t in range(problem.t)
        }

        for p in range(problem.p):
            for t in range(problem.t):
                delivered = model.sum(
                    quantity[p, v, i, t]
                    for v in range(problem.v)
                    for i in range(1, problem.i)
                )
                previous = (
                    problem.I_p_i_0[p][0]
                    if t == 0
                    else inventory[p, 0, t - 1]
                )
                model.add_constraint(
                    x[p, t] + previous - delivered == inventory[p, 0, t],
                    ctname=f"relax_plant_balance_{p}_{t}",
                )

                for i in range(1, problem.i):
                    received = model.sum(
                        quantity[p, v, i, t] for v in range(problem.v)
                    )
                    previous_customer = (
                        problem.I_p_i_0[p][i]
                        if t == 0
                        else inventory[p, i, t - 1]
                    )
                    model.add_constraint(
                        received
                        + previous_customer
                        - problem.d_p_i_t[p][i - 1][t]
                        == inventory[p, i, t],
                        ctname=f"relax_customer_balance_{p}_{i}_{t}",
                    )

        for t in range(problem.t):
            model.add_constraint(
                model.sum(problem.b_p[p] * x[p, t] for p in range(problem.p))
                <= problem.B,
                ctname=f"relax_production_capacity_{t}",
            )

        for p in range(problem.p):
            for i in range(problem.i):
                for t in range(problem.t):
                    model.add_constraint(
                        inventory[p, i, t] <= problem.U_p_i[p][i],
                        ctname=f"relax_inventory_capacity_{p}_{i}_{t}",
                    )

        return model, x, inventory, quantity

    @staticmethod
    def _value(solution, variable):
        value = solution.get_value(variable)
        if value is None or not math.isfinite(float(value)):
            raise RuntimeError("PL relaxado retornou um valor não finito")
        return float(value)

    def solve_bounds(self, include_base=False):
        lower = np.zeros((self.problem.p, self.problem.t), dtype=float)
        upper = np.zeros((self.problem.p, self.problem.t), dtype=float)

        model, x, _, _ = self._build_model()
        try:
            self._apply_time_limit(model)
            for p in range(self.problem.p):
                for t in range(self.problem.t):
                    model.minimize(x[p, t])
                    lower[p, t] = self._solve_model_value(model, x[p, t])
                    model.maximize(x[p, t])
                    upper[p, t] = self._solve_model_value(model, x[p, t])
        finally:
            model.end()

        result = {"lower": lower, "upper": upper}
        if include_base:
            model, x, inventory, quantity = self._build_model()
            try:
                self._apply_time_limit(model)
                model.minimize(model.sum(x[p, t] for p in range(self.problem.p) for t in range(self.problem.t)))
                solution = model.solve(log_output=False)
                if solution is None:
                    raise RuntimeError("PL relaxado da solução-base sem solução")
                result["base"] = self._extract_solution(solution, x, inventory, quantity)
            finally:
                model.end()
        return result

    def _apply_time_limit(self, model):
        if self.time_limit is not None:
            model.set_time_limit(float(self.time_limit))

    def _solve_model_value(self, model, variable):
        solution = model.solve(log_output=False)
        if solution is None:
            status = model.solve_details.status if model.solve_details else "desconhecido"
            raise RuntimeError(f"PL relaxado sem solução ({status})")
        return self._value(solution, variable)

    def _extract_solution(self, solution, x, inventory, quantity):
        result = {
            "X": np.zeros((self.problem.p, self.problem.t), dtype=float),
            "I": np.zeros((self.problem.p, self.problem.i, self.problem.t), dtype=float),
            "Q": np.zeros(
                (self.problem.p, self.problem.v, self.problem.i, self.problem.t),
                dtype=float,
            ),
        }
        for key, variable in x.items():
            result["X"][key] = self._value(solution, variable)
        for key, variable in inventory.items():
            result["I"][key] = self._value(solution, variable)
        for key, variable in quantity.items():
            result["Q"][key] = self._value(solution, variable)
        return result
