#################################################################################################
# Multi Product Prodction Routing Problem
# Copyright 2024 Mateus Chacon

# Este programa é um software livre, você pode redistribuí-lo e/ou modificá-lo
# sob os termos da Licença Pública Geral GNU como publicada pela Fundação do Software Livre (FSF),
# na versão 3 da Licença, ou (a seu critério) qualquer versão posterior.

# Este programa é distribuído na esperança de que possa ser útil, mas SEM NENHUMA GARANTIA,
# e sem uma garantia implícita de ADEQUAÇÃO a qualquer MERCADO ou APLICAÇÃO EM PARTICULAR.

import time

import numpy as np
from config import Config

# Veja a Licença Pública Geral GNU para mais detalhes
#################################################################################################
from docplex.mp.model import Model
from docplex.mp.relax_linear import LinearRelaxer
from src.log.Logger import Logger
from src.helpers.SolverTelemetry import get_config
from src.solvers.RoundedCapacitySeparation import (
    build_obligatory_demands,
    separate_cumulative_cuts,
)
from src.solvers._solver_common import (
    ProblemData,
    compute_delivery_upper_bounds,
    compute_inventory_upper_bounds,
    compute_production_upper_bounds,
)


class MultProductProdctionRoutingProblem:

    def __init__(
        self,
        map,
        dir,
        log: Logger,
        start,
        symmetry_breaking_hc1=None,
        coelho_inequalities=None,
        positive_only_deviations=None,
    ):
        self.log: Logger = log
        self.log.debug(">> Iniciando MultProductProdctionRoutingProblem.")
        self.model = Model(name="Multi_Product_Prodction_Routing_Problem")
        self.log.debug(">> Iniciado model.")
        self.j = 5
        self.p = map["num_products"]  ##Products
        self.i = map["num_customers"] + 1  ##Customers
        self.k = map["num_customers"] + 1  ##Customers
        self.t = map["num_periods"]  ##Periods
        self.v = map["num_vehicles"]  ##Vehicles
        self.B = map["B"]  ##Production capacity;
        self.b_p = map["b_p"]  ##Time required to produce item 𝑝;
        self.c_p = map["c_p"]  ##Production cost of item 𝑝;
        self.s_p = map["s_p"]  ##Setup cost of item 𝑝;
        self.M = map["M"]  ##Big number
        self.U_p_i = map["U_pi"]  ##Maximum inventory upper bound of item 𝑝 at site i;
        self.I_p_i_0 = map["I_pi0"]  ##Initial Inventory of item 𝑝 at site 𝑖;
        self.h_p_i = map["h_pi"]  ##Inventory cost of item 𝑝 at site 𝑖;
        self.C = map["C"]  ##Vehicle capacity;
        self.f = map["f"]  ##Fixed transportation cost;
        self.a_i_k = map[
            "a_ik"
        ]  ##Transportation cost for traveling from node 𝑖 to node k;
        self.d_p_i_t = map["d_pit"]  ##Demand of item 𝑝 at customer 𝑖 in period 𝑡.
        self.weight = map["weight"]  ##Weight of the objective function.
        self.targets = map.get("targets")
        self.model.X_p_t = {}  ##Quantity of item 𝑝 produced in period 𝑡.
        self.model.Y_p_t = {}  ##1, if item 𝑝 is produced in period 𝑡; or 0, otherwise.
        self.model.I_p_i_t = {}  ##Inventory of item 𝑝 at site 𝑖 in the end of period 𝑡.
        self.model.Z_v_i_k_t = (
            {}
        )  ##1, if vehicle v travels along edge (i,k) in period t; or 0, atherwise.
        self.model.R_p_v_i_k_t = (
            {}
        )  ##Quantity of item 𝑝 transported by vehicle 𝑣 on edge (𝑖, 𝑘) in period 𝑡;
        self.model.Q_p_v_i_t = (
            {}
        )  ##Quantity of item 𝑝 delivered by vehicle 𝑣 to customer 𝑖 in period 𝑡.
        self.dir = dir
        self.time = 0
        self.solCount = 0
        self.relaxedModelObjVal = 0
        self.objBound = 0
        self.nodeCount = 0
        self.nodesRemaining = 0
        self.start = start
        self.symmetry_breaking_hc1 = symmetry_breaking_hc1
        self.coelho_inequalities = coelho_inequalities
        if positive_only_deviations is None:
            positive_only_deviations = Config.get_nested(
                "solver", "goal_programming", "positive_only_deviations", default=True
            )
        self.positive_only_deviations = bool(positive_only_deviations)
        self.strengthened_bounds = bool(map.get("strengthened_bounds", True))
        problem_data = ProblemData.from_map(map)
        if self.strengthened_bounds:
            self.production_upper_bounds = compute_production_upper_bounds(problem_data)
        else:
            self.production_upper_bounds = np.full(
                (self.p, self.t), float(self.M), dtype=float
            )
        self.delivery_upper_bounds = compute_delivery_upper_bounds(problem_data)
        if self.strengthened_bounds:
            self.inventory_upper_bounds = compute_inventory_upper_bounds(problem_data)
        else:
            self.inventory_upper_bounds = np.broadcast_to(
                np.asarray(self.U_p_i, dtype=float)[:, :, None],
                (self.p, self.i, self.t),
            ).copy()
        self.M_p = np.minimum(
            float(self.M),
            np.asarray(self.d_p_i_t, dtype=float).sum(axis=(1, 2)),
        )
        self.alpha = map["alpha"] if "alpha" in map else None
        self.telemetry_config = get_config(Config)
        self._telemetry_started_at = None
        self._telemetry_events = []
        self._last_bound_progress_seconds = None
        self._last_bound_progress_value = None
        self._first_feasible_seconds = None
        self._gap_target_seconds = None
        self.rounded_capacity_config = Config.get_nested(
            "solver", "rounded_capacity_inequalities", default={}
        ) or {}
        self._rounded_capacity_z_indices = []
        self._rounded_capacity_cut_rows = {}
        self._rounded_capacity_obligatory_demands = None
        self._rounded_capacity_stats = {
            "callback_calls": 0,
            "separator_calls": 0,
            "candidate_evaluations": 0,
            "cuts_found": 0,
            "cuts_added": 0,
            "cuts_rejected": 0,
            "duplicate_cuts": 0,
            "callback_errors": 0,
            "separator_seconds": 0.0,
            "max_violation": 0.0,
        }
        self._rounded_capacity_callback_registered = False
        self.log.debug(">> Finalizado MultProductProdctionRoutingProblem.")

    def createDecisionVariables(self):
        self.log.debug(">> Iniciando createDecisionVariables.")
        for p in range(self.p):
            for t in range(self.t):
                self.model.X_p_t[p, t] = self.model.continuous_var(lb=0, name=f"X_{p}_{t}")
                self.model.Y_p_t[p, t] = self.model.binary_var(name=f"Y_{p}_{t}")
            for i in range(self.i):
                for t in range(self.t):
                    self.model.I_p_i_t[p, i, t] = self.model.continuous_var(
                        lb=0, name=f"I_{p}_{i}_{t}"
                    )
            for v in range(self.v):
                for i in range(self.i):
                    for k in range(self.k):
                        if i == k:
                            continue
                        for t in range(self.t):
                            self.model.R_p_v_i_k_t[p, v, i, k, t] = (
                                self.model.continuous_var(
                                    lb=0, name=f"R_{p}_{v}_{i}_{k}_{t}"
                                )
                            )
                for i in range(self.i):
                    for t in range(self.t):
                        self.model.Q_p_v_i_t[p, v, i, t] = self.model.continuous_var(
                            lb=0, name=f"Q_{p}_{v}_{i}_{t}"
                        )
        for v in range(self.v):
            for i in range(self.i):
                for k in range(self.k):
                    if i == k:
                        continue
                    for t in range(self.t):
                        self.model.Z_v_i_k_t[v, i, k, t] = self.model.binary_var(
                            name=f"Z_{v}_{i}_{k}_{t}"
                        )
        self.positive = self.model.continuous_var_dict(
            keys=((j, t) for j in range(self.j) for t in range(self.t)), name=f"p"
        )
        self.negative = None
        if not self.positive_only_deviations:
            self.negative = self.model.continuous_var_dict(
                keys=((j, t) for j in range(self.j) for t in range(self.t)), name=f"n"
            )
        self.lambda_ = self.model.continuous_var(name="lambda")

    def startVariables(self):
        warm_start = self.model.new_solution()
        for p in range(self.p):
            for t in range(self.t):
                warm_start.add_var_value(
                    self.model.X_p_t[p, t], float(self.start["variables"]["X"][p][t])
                )
                warm_start.add_var_value(
                    self.model.Y_p_t[p, t], float(self.start["variables"]["Y"][p][t])
                )
            for i in range(self.i):
                for t in range(self.t):
                    warm_start.add_var_value(
                        self.model.I_p_i_t[p, i, t],
                        float(self.start["variables"]["I"][p][i][t]),
                    )
            for v in range(self.v):
                for i in range(self.i):
                    for k in range(self.k):
                        if i == k:
                            continue
                        for t in range(self.t):
                            warm_start.add_var_value(
                                self.model.R_p_v_i_k_t[p, v, i, k, t],
                                float(self.start["variables"]["R"][p][v][i][k][t]),
                            )
                for i in range(self.i):
                    for t in range(self.t):
                        warm_start.add_var_value(
                            self.model.Q_p_v_i_t[p, v, i, t],
                            float(self.start["variables"]["Q"][p][v][i][t]),
                        )
        for v in range(self.v):
            for i in range(self.i):
                for k in range(self.k):
                    if i == k:
                        continue
                    for t in range(self.t):
                        warm_start.add_var_value(
                            self.model.Z_v_i_k_t[v, i, k, t],
                            float(self.start["variables"]["Z"][v][i][k][t]),
                        )
        self.model.add_mip_start(warm_start)

    def _adjust_targets(self):
        self.new_targets = {}
        for t in range(self.t):
            self.new_targets[t] = self.targets[t].copy()

        for k in [
            "f1_target",
            "f2_target",
            "f3_target",
            "f4_target",
            "f5_target",
        ]:
            values_k = [self.targets[t][k] for t in range(self.t)]
            if values_k:  # Only calculate mean if there are non-zero values
                mean_val = np.mean(values_k)
                for t in range(self.t):
                    if self.targets[t][k] == 0:
                        self.log.debug(
                            f">> Adjusting target {t}_{k} from {self.targets[t][k]} to {mean_val}"
                        )
                        self.new_targets[t][k] = mean_val

    def crateObjectiveFunction(self):
        objExpr_1 = [
            self.model.sum(self.c_p[p] * self.model.X_p_t[p, t] for p in range(self.p))
            for t in range(self.t)
        ]
        self.f1 = objExpr_1

        objExpr_2 = [
            self.model.sum(self.s_p[p] * self.model.Y_p_t[p, t] for p in range(self.p))
            for t in range(self.t)
        ]
        self.f2 = objExpr_2

        objExpr_3 = [
            self.model.sum(
                self.h_p_i[p][i] * self.model.I_p_i_t[p, i, t]
                for p in range(self.p)
                for i in range(self.i)
            )
            for t in range(self.t)
        ]
        self.f3 = objExpr_3

        objExpr_4 = [
            self.model.sum(
                self.f * self.model.Z_v_i_k_t[v, 0, k, t]
                for v in range(self.v)
                for k in range(1, self.k)
            )
            for t in range(self.t)
        ]
        self.f4 = objExpr_4

        objExpr_5 = [
            self.model.sum(
                self.a_i_k[i][k] * self.model.Z_v_i_k_t[v, i, k, t]
                for v in range(self.v)
                for i in range(self.i)
                for k in range(self.k)
                if i != k
            )
            for t in range(self.t)
        ]
        self.f5 = objExpr_5

        objExpr = (
            self.weight[0] * sum(self.f1)
            + self.weight[1] * sum(self.f2)
            + self.weight[2] * sum(self.f3)
            + self.weight[3] * sum(self.f4)
            + self.weight[4] * sum(self.f5)
        )
        if Config.get_nested("postprocessing", "build_target"):
            self.log.debug(">> FO build_target.")
            self.model.minimize(objExpr)
        else:
            if Config.get_nested("solver", "multiobjective"):
                self._adjust_targets()
                self.log.debug(">> FO multiobjective.")
                self.model.minimize(
                    self.alpha * self.lambda_
                    + self.model.sum(
                        + (1 - self.alpha)
                        * (self.weight[0] * self.positive[0, t])
                        / self.new_targets[t]["f1_target"]
                        + (1 - self.alpha)
                        * (self.weight[1] * self.positive[1, t])
                        / self.new_targets[t]["f2_target"]
                        + (1 - self.alpha)
                        * (self.weight[2] * self.positive[2, t])
                        / self.new_targets[t]["f3_target"]
                        + (1 - self.alpha)
                        * (self.weight[3] * self.positive[3, t])
                        / self.new_targets[t]["f4_target"]
                        + (1 - self.alpha)
                        * (self.weight[4] * self.positive[4, t])
                        / self.new_targets[t]["f5_target"]
                        for t in range(self.t)
                    )
                )
            else:
                self.log.debug(">> FO singleobjective.")
                self.model.minimize(
                    self.model.sum(self.f1)
                    + self.model.sum(self.f2)
                    + self.model.sum(self.f3)
                    + self.model.sum(self.f4)
                    + self.model.sum(self.f5)
                )

    def createEstablishInvetoryBalanceAtPlant(self):
        for p in range(self.p):
            for t in range(self.t):
                r1 = self.model.sum(
                    self.model.Q_p_v_i_t[p, v, i, t]
                    for v in range(self.v)
                    for i in range(1, self.i)
                )
                if t == 0:
                    self.model.add_constraint(
                        self.model.X_p_t[p, t] + self.I_p_i_0[p][0] - r1
                        == self.model.I_p_i_t[p, 0, t],
                        ctname=f"EQ_2_p_{p}_t_{t}",
                    )
                else:
                    self.model.add_constraint(
                        self.model.X_p_t[p, t] + self.model.I_p_i_t[p, 0, t - 1] - r1
                        == self.model.I_p_i_t[p, 0, t],
                        ctname=f"EQ_2_p_{p}_t_{t}",
                    )

    def creteInventoryBalancingInventoryCustomers(self):
        for p in range(self.p):
            for i in range(1, self.i):
                for t in range(self.t):
                    r2 = self.model.sum(
                        self.model.Q_p_v_i_t[p, v, i, t] for v in range(self.v)
                    )
                    if t == 0:
                        self.model.add_constraint(
                            r2 + self.I_p_i_0[p][i] - self.d_p_i_t[p][i - 1][t]
                            == self.model.I_p_i_t[p, i, t],
                            ctname=f"EQ_3_p_{p}_i_{i}_t_{t}",
                        )
                    else:
                        self.model.add_constraint(
                            r2
                            + self.model.I_p_i_t[p, i, t - 1]
                            - self.d_p_i_t[p][i - 1][t]
                            == self.model.I_p_i_t[p, i, t],
                            ctname=f"EQ_3_p_{p}_i_{i}_t_{t}",
                        )

    def createPlantsMaximum(self):
        for t in range(self.t):
            r3 = self.model.sum(
                self.b_p[p] * self.model.X_p_t[p, t] for p in range(self.p)
            )
            self.model.add_constraint(r3 <= self.B, ctname=f"EQ_4_t_{t+1}")

    def createRelationshipBetweenProduction(self):
        for p in range(self.p):
            for t in range(self.t):
                upper_bound = float(self.production_upper_bounds[p, t])
                self.model.add_constraint(
                    self.model.X_p_t[p, t] <= upper_bound * self.model.Y_p_t[p, t],
                    ctname=f"EQ_5_p_{p}_t_{t}",
                )

    def createDelimitMaximumCapacityItemsAtPlant(self):
        for p in range(self.p):
            for i in range(self.i):
                for t in range(self.t):
                    self.model.add_constraint(
                        self.model.I_p_i_t[p, i, t]
                        <= float(self.inventory_upper_bounds[p, i, t]),
                        ctname=f"EQ_6_p_{p}_i_{i}_t_{t}",
                    )

    def createVehiclePreventTransshipmentIntermediateNodes(self):
        for p in range(self.p):
            for v in range(self.v):
                for k in range(1, self.k):
                    for t in range(self.t):
                        r7_a = self.model.sum(
                            self.model.R_p_v_i_k_t[p, v, i, k, t]
                            for i in range(self.i)
                            if k != i
                        )
                        r7_b = self.model.sum(
                            self.model.R_p_v_i_k_t[p, v, k, l, t]
                            for l in range(self.i)
                            if k != l
                        )
                        self.model.add_constraint(
                            r7_a - r7_b == self.model.Q_p_v_i_t[p, v, k, t],
                            ctname=f"EQ_7_p_{p}_v_{v}_k_{k}_t_{t}",
                        )

    def createEliminationSubroutes(self):
        for p in range(self.p):
            for t in range(self.t):
                r8_a = self.model.sum(
                    self.model.R_p_v_i_k_t[p, v, 0, k, t]
                    for v in range(self.v)
                    for k in range(1, self.k)
                )
                r8_b = self.model.sum(
                    self.model.R_p_v_i_k_t[p, v, i, 0, t]
                    for v in range(self.v)
                    for i in range(1, self.i)
                )
                r8_c = self.model.sum(
                    self.model.Q_p_v_i_t[p, v, l, t]
                    for v in range(self.v)
                    for l in range(1, self.i)
                )
                self.model.add_constraint(
                    r8_a - r8_b == r8_c, ctname=f"EQ_8_p_{p}_t_{t}"
                )

    def createVehicleLoadCapacityDelimited(self):
        for v in range(self.v):
            for i in range(self.i):
                for k in range(self.k):
                    for t in range(self.t):
                        if i != k:
                            r9 = self.model.sum(
                                self.model.R_p_v_i_k_t[p, v, i, k, t]
                                for p in range(self.p)
                            )
                            self.model.add_constraint(
                                r9 <= self.C * self.model.Z_v_i_k_t[v, i, k, t],
                                ctname=f"EQ_9_v_{v}_i_{i}_k_{k}_t_{t}",
                            )

    def createImposeMostOneRouteEachVehicle(self):
        for v in range(self.v):
            for t in range(self.t):
                r10 = self.model.sum(
                    self.model.Z_v_i_k_t[v, 0, k, t] for k in range(1, self.k)
                )
                self.model.add_constraint(r10 <= 1, ctname=f"EQ_10_v_{v}_t_{t}")

    def createEnsureRoutesOnlyPlant(self):
        for v in range(self.v):
            for k in range(self.k):
                for t in range(self.t):
                    r11_a = self.model.sum(
                        self.model.Z_v_i_k_t[v, i, k, t]
                        for i in range(self.i)
                        if k != i
                    )
                    r11_b = self.model.sum(
                        self.model.Z_v_i_k_t[v, k, l, t]
                        for l in range(self.i)
                        if k != l
                    )
                    self.model.add_constraint(
                        r11_a - r11_b == 0, ctname=f"EQ_11_v_{v}_k_{k}_t_{t}"
                    )

    def createVehicleMostVisitCustomerEachPeriod(self):
        for k in range(1, self.k):
            for t in range(self.t):
                r12 = self.model.sum(
                    self.model.Z_v_i_k_t[v, i, k, t]
                    for v in range(self.v)
                    for i in range(self.i)
                    if k != i
                )
                self.model.add_constraint(r12 <= 1, ctname=f"EQ_12_k_{k}_t_{t}")

    def createVehicleVisitDeliveryBounds(self):
        for v in range(self.v):
            for i in range(1, self.i):
                for t in range(self.t):
                    visit = self.model.sum(
                        self.model.Z_v_i_k_t[v, i, k, t]
                        for k in range(self.k)
                        if k != i
                    )
                    self.model.add_constraint(
                        self.model.sum(
                            self.model.Q_p_v_i_t[p, v, i, t]
                            for p in range(self.p)
                        )
                        <= self.C * visit,
                        ctname=f"VISIT_CAP_Z_v_{v}_i_{i}_t_{t}",
                    )
                    for p in range(self.p):
                        q_upper = float(self.delivery_upper_bounds[p, i, t])
                        self.model.add_constraint(
                            self.model.Q_p_v_i_t[p, v, i, t]
                            <= q_upper * visit,
                            ctname=f"VISIT_Q_Z_p_{p}_v_{v}_i_{i}_t_{t}",
                        )
        return True

    def createCoelhoValidInequalities(self):
        """Add logical inequalities (15)-(17) from Coelho and Laporte.

        The article uses undirected edge variables x and visit variables y.
        This model stores directed arcs in Z, so the same logic is written
        using projected visit and vehicle-use expressions over Z.
        """
        coelho_inequalities = self.coelho_inequalities
        if coelho_inequalities is None:
            coelho_inequalities = Config.get_nested(
                "solver", "coelho_inequalities", default=False
            )
        if not coelho_inequalities:
            return False

        for v in range(self.v):
            for t in range(self.t):
                vehicle_active = self.model.sum(
                    self.model.Z_v_i_k_t[v, 0, k, t] for k in range(1, self.k)
                )

                # Eq. (15): x_0i <= 2 y_i.
                for i in range(1, self.i):
                    customer_visit = self.model.sum(
                        self.model.Z_v_i_k_t[v, i, k, t]
                        for k in range(self.k)
                        if k != i
                    )
                    self.model.add_constraint(
                        self.model.Z_v_i_k_t[v, 0, i, t] <= 2 * customer_visit,
                        ctname=f"COELHO_15_v_{v}_i_{i}_t_{t}",
                    )

                # Eq. (16): x_ij <= y_i.
                for i in range(self.i):
                    origin_visit = (
                        vehicle_active
                        if i == 0
                        else self.model.sum(
                            self.model.Z_v_i_k_t[v, i, k, t]
                            for k in range(self.k)
                            if k != i
                        )
                    )
                    for k in range(self.k):
                        if i == k:
                            continue
                        self.model.add_constraint(
                            self.model.Z_v_i_k_t[v, i, k, t] <= origin_visit,
                            ctname=f"COELHO_16_v_{v}_i_{i}_k_{k}_t_{t}",
                        )

                # Eq. (17): y_i <= y_0.
                for i in range(1, self.i):
                    customer_visit = self.model.sum(
                        self.model.Z_v_i_k_t[v, i, k, t]
                        for k in range(self.k)
                        if k != i
                    )
                    self.model.add_constraint(
                        customer_visit <= vehicle_active,
                        ctname=f"COELHO_17_v_{v}_i_{i}_t_{t}",
                    )
        return True

    def createVehicleSymmetryBreaking(self):
        """Remove equivalent vehicle-label permutations from the MIP model.

        The vehicle indices are interchangeable in the current formulation. VC
        canonicalizes vehicle activation, and HC1 additionally requires a
        higher-index vehicle to serve a customer only when the preceding vehicle
        serves at least one lower-index customer in the same period.
        """
        symmetry_breaking_hc1 = self.symmetry_breaking_hc1
        if symmetry_breaking_hc1 is None:
            symmetry_breaking_hc1 = Config.get_nested(
                "solver", "symmetry_breaking", "hc1", default=False
            )
        if not symmetry_breaking_hc1:
            return False

        for v in range(1, self.v):
            for t in range(self.t):
                current_vehicle_active = self.model.sum(
                    self.model.Z_v_i_k_t[v, 0, k, t] for k in range(1, self.k)
                )
                previous_vehicle_active = self.model.sum(
                    self.model.Z_v_i_k_t[v - 1, 0, k, t]
                    for k in range(1, self.k)
                )
                self.model.add_constraint(
                    current_vehicle_active <= previous_vehicle_active,
                    ctname=f"SB_VC_v_{v}_t_{t}",
                )

            for i in range(1, self.i):
                for t in range(self.t):
                    current_vehicle_visit = self.model.sum(
                        self.model.Z_v_i_k_t[v, i, k, t]
                        for k in range(self.k)
                        if k != i
                    )
                    previous_vehicle_lower_customer_visit = self.model.sum(
                        self.model.Z_v_i_k_t[v - 1, j, k, t]
                        for j in range(1, i)
                        for k in range(self.k)
                        if k != j
                    )
                    self.model.add_constraint(
                        current_vehicle_visit <= previous_vehicle_lower_customer_visit,
                        ctname=f"SB_HC1_v_{v}_i_{i}_t_{t}",
                    )
        return True

    def createGoalProgrammingRestrictions(self):
        for t in range(self.t):
            objectives = self.f1, self.f2, self.f3, self.f4, self.f5
            target_keys = "f1_target", "f2_target", "f3_target", "f4_target", "f5_target"
            if self.positive_only_deviations:
                self.model.add_constraints(
                    self.positive[j, t] >= objectives[j][t] - self.targets[t][target_keys[j]]
                    for j in range(self.j)
                )
            else:
                self.model.add_constraints(
                    objectives[j][t]
                    + self.negative[j, t]
                    - self.positive[j, t]
                    == self.targets[t][target_keys[j]]
                    for j in range(self.j)
                )

    def createEpsilonRestricted(self):
        for t in range(self.t):
            self.model.add_constraints(
                [
                    self.weight[0] * self.positive[0, t]
                    <= self.lambda_ * self.new_targets[t]["f1_target"],
                    self.weight[1] * self.positive[1, t]
                    <= self.lambda_ * self.new_targets[t]["f2_target"],
                    self.weight[2] * self.positive[2, t]
                    <= self.lambda_ * self.new_targets[t]["f3_target"],
                    self.weight[3] * self.positive[3, t]
                    <= self.lambda_ * self.new_targets[t]["f4_target"],
                    self.weight[4] * self.positive[4, t]
                    <= self.lambda_ * self.new_targets[t]["f5_target"],
                ]
            )

    def outModel(self):
        self.model.export_as_lp(f"{self.dir}modelo.lp")

    def getResults(self):
        if self.solCount == 0:
            return (
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                0,
                0,
                self.time,
                None,
                self.solCount,
                self.relaxedModelObjVal,
                self.nodeCount,
                self.objBound,
                None,
            )

        epsilon = None
        try:
            if Config.get_nested("solver", "multiobjective"):
                epsilon = float(self.lambda_.solution_value)
        except Exception:
            epsilon = None

        P = []
        try:
            for t in range(self.t):
                p_t = []
                for j in range(self.j):
                    p_t.append(float(self.positive[j, t].solution_value))
                P.append(p_t)
        except Exception:
            P = []

        Z = []
        for t in range(self.t):
            v_list = []
            for v in range(self.v):
                i_list = []
                for i in range(self.i):
                    k_list = []
                    for k in range(self.k):
                        if i == k:
                            k_list.append(0.0)
                            continue
                        variable = abs(
                            self.model.get_var_by_name(
                                f"Z_{v}_{i}_{k}_{t}"
                            ).solution_value
                        )
                        k_list.append(variable)
                    i_list.append(k_list)
                v_list.append(i_list)
            Z.append(v_list)
        Y = []
        for t in range(self.t):
            p_list_y = []
            for p in range(self.p):
                variable = abs(self.model.get_var_by_name(f"Y_{p}_{t}").solution_value)
                p_list_y.append(variable)
            Y.append(p_list_y)

        X = []
        for t in range(self.t):
            p_list_x = []
            for p in range(self.p):
                variable = self.model.get_var_by_name(f"X_{p}_{t}").solution_value
                p_list_x.append(variable)
            X.append(p_list_x)

        I = []
        for t in range(self.t):
            p_list_i = []
            for i in range(self.i):
                i_list_i = []
                for p in range(self.p):
                    variable = self.model.get_var_by_name(
                        f"I_{p}_{i}_{t}"
                    ).solution_value
                    i_list_i.append(variable)
                p_list_i.append(i_list_i)
            I.append(p_list_i)
        R = []
        for t in range(self.t):
            t_list = []
            for v in range(self.v):
                v_list = []
                for p in range(self.p):
                    p_list = []
                    for i in range(self.i):
                        i_list = []
                        for k in range(self.k):
                            if i == k:
                                i_list.append(0.0)
                                continue
                            i_list.append(
                                float(
                                    self.model.get_var_by_name(
                                        f"R_{p}_{v}_{i}_{k}_{t}"
                                    ).solution_value
                                )
                            )
                        p_list.append(i_list)
                    v_list.append(p_list)
                t_list.append(v_list)
            R.append(t_list)
        Q = []
        for t in range(self.t):
            t_list = []
            for v in range(self.v):
                v_list = []
                for p in range(self.p):
                    p_list = []
                    for i in range(self.i):
                        variable = self.model.get_var_by_name(
                            f"Q_{p}_{v}_{i}_{t}"
                        ).solution_value
                        p_list.append(variable)
                    v_list.append(p_list)
                t_list.append(v_list)
            Q.append(t_list)
        return (
            Z,
            X,
            Y,
            I,
            R,
            Q,
            P,
            self.model.objective_value,
            self.model.solve_details.mip_relative_gap,
            self.time,
            epsilon,
            self.solCount,
            self.relaxedModelObjVal,
            self.nodeCount,
            self.objBound,
            getattr(self, "new_targets", None),
        )

    def terminate(self):
        model = getattr(self, "model", None)
        if model is not None:
            model.end()
            self.model = None
        if hasattr(self, "solution"):
            self.solution = None

    def generteRelax(self, REPLACE_MODEL=True):
        # Create a linear relaxation of the model using docplex
        linear_relaxer = LinearRelaxer()
        relaxed = linear_relaxer.linear_relaxation(self.model.clone())
        if relaxed is None:
            self.log.error("Relaxation not generated")
            return
        if REPLACE_MODEL:
            self.log.debug("Relaxation model replaced")
            original_model = self.model
            self.model = relaxed
            original_model.end()

        try:
            solution = relaxed.solve()
            if solution:
                self.relaxedModelObjVal = solution.objective_value
            else:
                self.relaxedModelObjVal = 0
        finally:
            if not REPLACE_MODEL:
                relaxed.end()

    def _get_cplex_progress_value(self, method_name):
        try:
            progress = self.model.get_cplex().solution.progress
            value = getattr(progress, method_name)()
            return int(value) if value is not None else None
        except (AttributeError, TypeError, ValueError, RuntimeError):
            return None

    def _update_node_progress(self, processed=None, remaining=None):
        if processed is not None:
            self.nodeCount = max(self.nodeCount, int(processed))
        if remaining is not None:
            self.nodesRemaining = int(remaining)

    def processInformationsSolver(self):
        details = getattr(self.model, "solve_details", None)
        if details:
            self.solCount = 1 if self.model.solution else 0
            self.objBound = (
                details.best_bound if hasattr(details, "best_bound") else 0
            )

            detail_nodes = getattr(details, "nb_nodes_processed", None)
            progress_nodes = self._get_cplex_progress_value(
                "get_num_nodes_processed"
            )
            self._update_node_progress(processed=progress_nodes)
            node_counts = [
                int(value)
                for value in (detail_nodes, progress_nodes)
                if value is not None and int(value) >= 0
            ]
            self.nodeCount = max(self.nodeCount, max(node_counts, default=0))
            progress_remaining = self._get_cplex_progress_value(
                "get_num_nodes_remaining"
            )
            self._update_node_progress(remaining=progress_remaining)
        else:
            self.solCount = 0
            self.objBound = 0
            self.nodeCount = 0
            self.nodesRemaining = 0

    def _record_mip_telemetry(self, elapsed_seconds, has_incumbent, relative_gap):
        if not self.telemetry_config["enabled"]:
            return
        if has_incumbent and self._first_feasible_seconds is None:
            self._first_feasible_seconds = float(elapsed_seconds)
            self._telemetry_events.append({"event": "first_feasible", "elapsed_seconds": float(elapsed_seconds), "source": "mip"})
        if relative_gap is not None and relative_gap <= self.telemetry_config["gap_target_relative"] and self._gap_target_seconds is None:
            self._gap_target_seconds = float(elapsed_seconds)
            self._telemetry_events.append({"event": "gap_target", "elapsed_seconds": float(elapsed_seconds), "gap": float(relative_gap)})

    def _record_bound_progress(
        self,
        elapsed_seconds,
        best_bound,
        incumbent_objective=None,
        relative_gap=None,
        nodes_processed=None,
        nodes_remaining=None,
    ):
        if not self.telemetry_config["enabled"] or best_bound is None:
            return
        try:
            best_bound = float(best_bound)
        except (TypeError, ValueError):
            return

        interval = self.telemetry_config.get("bound_progress_interval_seconds", 1.0)
        elapsed_seconds = float(elapsed_seconds)
        if (
            self._last_bound_progress_seconds is not None
            and elapsed_seconds - self._last_bound_progress_seconds < interval
            and best_bound == self._last_bound_progress_value
        ):
            return

        event = {
            "event": "bound_progress",
            "elapsed_seconds": elapsed_seconds,
            "best_bound": best_bound,
            "incumbent_objective": (
                float(incumbent_objective)
                if incumbent_objective is not None
                else None
            ),
            "relative_gap": (
                float(relative_gap) if relative_gap is not None else None
            ),
            "nodes_processed": (
                int(nodes_processed) if nodes_processed is not None else None
            ),
            "nodes_remaining": (
                int(nodes_remaining) if nodes_remaining is not None else None
            ),
            "is_root": nodes_processed == 0 and nodes_remaining == 1,
        }
        self._telemetry_events.append(event)
        self._last_bound_progress_seconds = elapsed_seconds
        self._last_bound_progress_value = best_bound

    def get_telemetry(self):
        details = getattr(self.model, "solve_details", None)
        status = str(getattr(details, "status", "unknown"))
        timed_out = "time limit" in status.lower()
        objective = None
        if getattr(self, "solution", None) is not None:
            try:
                objective = float(self.model.objective_value)
            except Exception:
                pass
        gap = getattr(details, "mip_relative_gap", None)
        self._record_mip_telemetry(self.time, objective is not None, gap)
        self._record_bound_progress(
            self.time,
            getattr(details, "best_bound", None),
            incumbent_objective=objective,
            relative_gap=gap,
            nodes_processed=self.nodeCount,
            nodes_remaining=self.nodesRemaining,
        )
        rounded_capacity = dict(self._rounded_capacity_stats)
        rounded_capacity["callback_registered"] = self._rounded_capacity_callback_registered
        separator_calls = rounded_capacity["separator_calls"]
        rounded_capacity["separator_average_seconds"] = (
            rounded_capacity["separator_seconds"] / separator_calls
            if separator_calls
            else 0.0
        )
        root_bounds = [
            event["best_bound"]
            for event in self._telemetry_events
            if event.get("event") == "bound_progress" and event.get("is_root")
        ]
        return {
            "strategy": "solver",
            "mip_seconds": float(self.time), "status": status, "timed_out": timed_out,
            "objective": objective, "best_bound": getattr(details, "best_bound", None),
            "root_bound": root_bounds[0] if root_bounds else None,
            "relative_gap": gap, "node_count": self.nodeCount, "solution_count": self.solCount,
            "nodes_remaining": self.nodesRemaining,
            "positive_only_deviations": self.positive_only_deviations,
            "strengthened_bounds": self.strengthened_bounds,
            "first_feasible_seconds": self._first_feasible_seconds,
            "gap_target_seconds": self._gap_target_seconds,
            "mip_events": list(self._telemetry_events), "pso_iterations": [],
            "rounded_capacity": rounded_capacity,
        }

    def _rounded_capacity_enabled(self):
        return bool(self.rounded_capacity_config.get("enabled", False))

    def _prepare_rounded_capacity_callback_data(self):
        if not self._rounded_capacity_enabled():
            return False

        self._rounded_capacity_obligatory_demands = build_obligatory_demands(
            self.d_p_i_t, self.I_p_i_0
        )
        cplex_model = self.model.get_cplex()
        index_by_name = {
            name: index for index, name in enumerate(cplex_model.variables.get_names())
        }
        self._rounded_capacity_z_indices = []
        for t in range(self.t):
            for v in range(self.v):
                for i in range(self.i):
                    for k in range(self.k):
                        if i == k:
                            continue
                        name = f"Z_{v}_{i}_{k}_{t}"
                        try:
                            index = index_by_name[name]
                        except KeyError as error:
                            raise RuntimeError(
                                f"Variavel de rota ausente no callback: {name}"
                            ) from error
                        self._rounded_capacity_z_indices.append((t, v, i, k, index))
        return True

    def _rounded_capacity_should_separate(self, node_count):
        if node_count == 0:
            return True
        max_non_root_node = int(
            self.rounded_capacity_config.get("max_non_root_node", 200)
        )
        frequency = int(self.rounded_capacity_config.get("node_frequency", 50))
        return node_count <= max_non_root_node and node_count % frequency == 0

    def _rounded_capacity_build_row(self, cut):
        selected = set(customer + 1 for customer in cut.customers)
        indices = []
        values = []
        for t, v, i, k, index in self._rounded_capacity_z_indices:
            if t > cut.tau:
                continue
            if (i in selected) != (k in selected):
                indices.append(index)
                values.append(1.0)
        return indices, values

    def _run_rounded_capacity_separator(self, callback, max_cuts):
        started_at = time.perf_counter()
        self._rounded_capacity_stats["separator_calls"] += 1
        try:
            indices = [item[4] for item in self._rounded_capacity_z_indices]
            values = callback.get_values(indices)
            z_values = np.zeros((self.t, self.v, self.i, self.k), dtype=float)
            for item, value in zip(self._rounded_capacity_z_indices, values):
                t, v, i, k, _ = item
                z_values[t, v, i, k] = float(value)

            separation_stats = {}
            cuts = separate_cumulative_cuts(
                z_values,
                self._rounded_capacity_obligatory_demands,
                self.C,
                max_cuts=max_cuts,
                min_violation=float(
                    self.rounded_capacity_config.get("min_violation", 1e-6)
                ),
                customer_node_offset=1,
                statistics=separation_stats,
            )
            self._rounded_capacity_stats["candidate_evaluations"] += int(
                separation_stats.get("candidate_evaluations", 0)
            )
            self._rounded_capacity_stats["cuts_found"] += len(cuts)
            for cut in cuts:
                self._rounded_capacity_stats["max_violation"] = max(
                    self._rounded_capacity_stats["max_violation"], cut.violation
                )
                if cut.key in self._rounded_capacity_cut_rows:
                    self._rounded_capacity_stats["duplicate_cuts"] += 1
                    continue
                row_indices, row_values = self._rounded_capacity_build_row(cut)
                if not row_indices:
                    self._rounded_capacity_stats["cuts_rejected"] += 1
                    continue
                self._rounded_capacity_cut_rows[cut.key] = (cut.rhs, cut.violation)
                import cplex

                callback.add(
                    cplex.SparsePair(ind=row_indices, val=row_values), "G", cut.rhs
                )
                self._rounded_capacity_stats["cuts_added"] += 1
        finally:
            self._rounded_capacity_stats["separator_seconds"] += (
                time.perf_counter() - started_at
            )

    def _install_rounded_capacity_callback(self):
        if not self._rounded_capacity_enabled():
            return False
        try:
            from cplex.callbacks import UserCutCallback

            self._prepare_rounded_capacity_callback_data()
            owner = self

            class RoundedCapacityCallback(UserCutCallback):
                def __call__(self):
                    owner._rounded_capacity_stats["callback_calls"] += 1
                    try:
                        node_count = int(self.get_num_nodes())
                        if not owner._rounded_capacity_should_separate(node_count):
                            return
                        max_cuts = int(
                            owner.rounded_capacity_config.get(
                                "max_cuts_per_callback", 20
                            )
                            if node_count == 0
                            else owner.rounded_capacity_config.get(
                                "max_cuts_per_non_root_callback", 3
                            )
                        )
                        owner._run_rounded_capacity_separator(self, max_cuts)
                    except Exception as error:
                        owner._rounded_capacity_stats["callback_errors"] += 1
                        owner.log.warning(
                            f"Separacao de capacidade arredondada ignorada: {error}"
                        )

            self.model.register_callback(RoundedCapacityCallback)
            self._rounded_capacity_callback_registered = True
            return True
        except Exception as error:
            self.log.warning(f"Callback de capacidade arredondada indisponivel: {error}")
            return False

    def _install_mip_telemetry_callback(self):
        try:
            from cplex.callbacks import MIPInfoCallback
            owner = self
            started_at = time.perf_counter()
            class TelemetryCallback(MIPInfoCallback):
                def __call__(self):
                    try:
                        owner._update_node_progress(
                            processed=self.get_num_nodes(),
                            remaining=self.get_num_remaining_nodes(),
                        )
                        if not owner.telemetry_config["enabled"]:
                            return
                        elapsed = time.perf_counter() - started_at
                        incumbent = self.has_incumbent()
                        gap = self.get_MIP_relative_gap() if incumbent else None
                        best_bound = self.get_best_objective_value()
                        incumbent_objective = (
                            self.get_incumbent_objective_value()
                            if incumbent
                            else None
                        )
                        owner._record_bound_progress(
                            elapsed,
                            best_bound,
                            incumbent_objective=incumbent_objective,
                            relative_gap=gap,
                            nodes_processed=self.get_num_nodes(),
                            nodes_remaining=self.get_num_remaining_nodes(),
                        )
                        owner._record_mip_telemetry(elapsed, incumbent, gap)
                    except Exception:
                        pass
            self.model.register_callback(TelemetryCallback)
        except Exception as error:
            self.log.warning(f"Telemetria MIP sem callback: {error}")

    def solver(self, numThreads=None, timeLimit=None):
        self.createDecisionVariables()

        self.log.debug(f"Variabes.start == {self.start['start']}")
        if self.start["start"] == True:
            self.startVariables()

        self.crateObjectiveFunction()
        self.log.debug("Objetivo criado")
        if Config.get_nested("solver", "multiobjective"):
            self.createGoalProgrammingRestrictions()
            self.createEpsilonRestricted()
        self.createEstablishInvetoryBalanceAtPlant()
        self.log.debug("Balanceamento estoque Planta criado")
        self.creteInventoryBalancingInventoryCustomers()
        self.log.debug("Balanceamento estoque Cliente criado")
        self.createPlantsMaximum()
        self.log.debug("Planta max criado")
        self.createRelationshipBetweenProduction()
        self.log.debug("Produção criado")
        self.createDelimitMaximumCapacityItemsAtPlant()
        self.log.debug("Capacidade maxima planta criado")
        self.createVehiclePreventTransshipmentIntermediateNodes()
        self.log.debug("Transshipment criado")
        self.createEliminationSubroutes()
        self.log.debug("Subrotas criado")
        self.createVehicleLoadCapacityDelimited()
        self.log.debug("Capacidade maxima veículo criado")
        self.createImposeMostOneRouteEachVehicle()
        self.log.debug("Max rota veículo criado")
        self.createEnsureRoutesOnlyPlant()
        self.log.debug("Rota somente entre plantas criado")
        self.createVehicleMostVisitCustomerEachPeriod()
        self.log.debug("Veículo visita cliente criado")
        if self.createVehicleVisitDeliveryBounds():
            self.log.debug("Bounds de entrega por visita em função de Z criados")
        if self.createCoelhoValidInequalities():
            self.log.debug("Desigualdades lógicas de Coelho (15)-(17) criadas")
        if self.createVehicleSymmetryBreaking():
            self.log.debug("Quebra de simetria VC + HC1 criada")
        # self.outModel()
        if Config.get_nested("postprocessing", "build_target"):
            target_use_relaxation = Config.get_nested(
                "relaxed_solution", "target_use_relaxation", default=True
            )
            if target_use_relaxation:
                self.generteRelax(
                    REPLACE_MODEL=Config.get_nested("relaxed_solution", "replace_model")
                )
                self.log.debug("Solução relaxada gerada para construção do target")
            else:
                self.log.debug("Construção do target usando o modelo MIP inteiro")
        else:
            self.log.debug("Solução não relaxada - usando modelo original")

        # Set parameters
        if timeLimit is not None:
            self.model.set_time_limit(timeLimit)
        if numThreads is not None:
            self.model.context.cplex_parameters.threads = numThreads

        start_time = time.time()
        self._telemetry_started_at = time.perf_counter()
        self._install_mip_telemetry_callback()
        if not Config.get_nested("postprocessing", "build_target", default=False):
            self._install_rounded_capacity_callback()
        if not Config.get_nested("relaxed_solution", "use"):
            self.solution = self.model.solve(log_output=False)
        end_time = time.time()
        self.time = end_time - start_time

        self.processInformationsSolver()
        elapsed = time.perf_counter() - self._telemetry_started_at
        details = getattr(self.model, "solve_details", None)
        self._record_mip_telemetry(elapsed, self.solCount > 0, getattr(details, "mip_relative_gap", None))
