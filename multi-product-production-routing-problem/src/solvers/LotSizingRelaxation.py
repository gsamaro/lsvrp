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

    def _build_model(
        self,
        production_lower_bounds=None,
        quantity_lower_bounds=None,
    ):
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

        for v in range(problem.v):
            for t in range(problem.t):
                model.add_constraint(
                    model.sum(
                        quantity[p, v, i, t]
                        for p in range(problem.p)
                        for i in range(1, problem.i)
                    )
                    <= problem.C,
                    ctname=f"relax_vehicle_delivery_capacity_{v}_{t}",
                )

        for p in range(problem.p):
            for i in range(problem.i):
                for t in range(problem.t):
                    model.add_constraint(
                        inventory[p, i, t] <= problem.U_p_i[p][i],
                        ctname=f"relax_inventory_capacity_{p}_{i}_{t}",
                    )

        if production_lower_bounds is not None:
            for p in range(problem.p):
                for t in range(problem.t):
                    model.add_constraint(
                        x[p, t] >= float(production_lower_bounds[p, t]),
                        ctname=f"relax_production_lower_bound_{p}_{t}",
                    )

        if quantity_lower_bounds is not None:
            for p in range(problem.p):
                for v in range(problem.v):
                    for i in range(problem.i):
                        for t in range(problem.t):
                            model.add_constraint(
                                quantity[p, v, i, t]
                                >= float(quantity_lower_bounds[p, v, i, t]),
                                ctname=f"relax_delivery_lower_bound_{p}_{v}_{i}_{t}",
                            )

        return model, x, inventory, quantity

    @staticmethod
    def _value(solution, variable):
        value = solution.get_value(variable)
        if value is None or not math.isfinite(float(value)):
            raise RuntimeError("PL relaxado retornou um valor não finito")
        return float(value)

    def solve_bounds(self, include_base=False):
        lower_result = self._solve_sum_model(minimize=True, label="lower")
        lower_solution = lower_result["solution"]
        upper_result = self._solve_sum_model(
            minimize=False,
            label="upper",
            production_lower_bounds=lower_solution["X"],
            quantity_lower_bounds=lower_solution["Q"],
        )

        result = {
            "lower": lower_result["objective"],
            "upper": upper_result["objective"],
            "lower_solution": lower_solution,
            "upper_solution": upper_result["solution"],
        }
        if include_base:
            result["base"] = lower_solution
        return result

    def _solve_sum_model(
        self,
        minimize,
        label,
        production_lower_bounds=None,
        quantity_lower_bounds=None,
    ):
        model, x, inventory, quantity = self._build_model(
            production_lower_bounds=production_lower_bounds,
            quantity_lower_bounds=quantity_lower_bounds,
        )
        try:
            self._apply_time_limit(model)
            objective = model.sum(
                quantity[p, v, i, t]
                for p in range(self.problem.p)
                for v in range(self.problem.v)
                for i in range(1, self.problem.i)
                for t in range(self.problem.t)
            )
            if minimize:
                model.minimize(objective)
            else:
                model.maximize(objective)
            solution = model.solve(log_output=False)
            if solution is None:
                status = model.solve_details.status if model.solve_details else "desconhecido"
                raise RuntimeError(f"PL relaxado sem solução para {label} ({status})")
            return {
                "objective": float(solution.objective_value),
                "solution": self._extract_solution(solution, x, inventory, quantity),
            }
        finally:
            model.end()

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
