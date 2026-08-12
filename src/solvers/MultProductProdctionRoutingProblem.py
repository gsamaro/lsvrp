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


class MultProductProdctionRoutingProblem:

    def __init__(self, map, dir, log: Logger, start):
        self.log: Logger = log
        self.log.info(">> Iniciando MultProductProdctionRoutingProblem.")
        self.model = Model(name="Multi_Product_Prodction_Routing_Problem")
        self.log.info(">> Iniciado model.")
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
        self.log: Logger = log
        self.start = start
        self.alpha = map["alpha"] if "alpha" in map else None
        self.telemetry_config = get_config(Config)
        self._telemetry_started_at = None
        self._telemetry_events = []
        self._first_feasible_seconds = None
        self._gap_target_seconds = None
        self.log.info(">> Finalizado MultProductProdctionRoutingProblem.")

    def createDecisionVariables(self):
        self.log.info(">> Iniciando createDecisionVariables.")
        for p in range(self.p):
            for t in range(self.t):
                self.model.X_p_t[p, t] = self.model.integer_var(lb=0, name=f"X_{p}_{t}")
                self.model.Y_p_t[p, t] = self.model.binary_var(name=f"Y_{p}_{t}")
            for i in range(self.i):
                for t in range(self.t):
                    self.model.I_p_i_t[p, i, t] = self.model.integer_var(
                        lb=0, name=f"I_{p}_{i}_{t}"
                    )
            for v in range(self.v):
                for i in range(self.i):
                    for k in range(self.k):
                        for t in range(self.t):
                            self.model.R_p_v_i_k_t[p, v, i, k, t] = (
                                self.model.integer_var(
                                    lb=0, name=f"R_{p}_{v}_{i}_{k}_{t}"
                                )
                            )
                for i in range(self.i):
                    for t in range(self.t):
                        self.model.Q_p_v_i_t[p, v, i, t] = self.model.integer_var(
                            lb=0, name=f"Q_{p}_{v}_{i}_{t}"
                        )
        for v in range(self.v):
            for i in range(self.i):
                for k in range(self.k):
                    for t in range(self.t):
                        self.model.Z_v_i_k_t[v, i, k, t] = self.model.binary_var(
                            name=f"Z_{v}_{i}_{k}_{t}"
                        )
        self.positive = self.model.continuous_var_dict(
            keys=((j, t) for j in range(self.j) for t in range(self.t)), name=f"p"
        )
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
                        self.log.info(
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
            self.log.info(">> FO build_target.")
            self.model.minimize(objExpr)
        else:
            if Config.get_nested("solver", "multiobjective"):
                self._adjust_targets()
                self.log.info(">> FO multiobjective.")
                self.model.minimize(
                    self.model.sum(
                        self.alpha * self.lambda_
                        + (1 - self.alpha)
                        * (self.weight[0] * self.positive[0, t])
                        / self.new_targets[t]["f1_target"]
                        + self.alpha * self.lambda_
                        + (1 - self.alpha)
                        * (self.weight[1] * self.positive[1, t])
                        / self.new_targets[t]["f2_target"]
                        + self.alpha * self.lambda_
                        + (1 - self.alpha)
                        * (self.weight[2] * self.positive[2, t])
                        / self.new_targets[t]["f3_target"]
                        + self.alpha * self.lambda_
                        + (1 - self.alpha)
                        * (self.weight[3] * self.positive[3, t])
                        / self.new_targets[t]["f4_target"]
                        + self.alpha * self.lambda_
                        + (1 - self.alpha)
                        * (self.weight[4] * self.positive[4, t])
                        / self.new_targets[t]["f5_target"]
                        for t in range(self.t)
                    )
                )
            else:
                self.log.info(">> FO singleobjective.")
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
                self.model.add_constraint(
                    self.model.X_p_t[p, t] <= self.M * self.model.Y_p_t[p, t],
                    ctname=f"EQ_5_p_{p}_t_{t}",
                )

    def createDelimitMaximumCapacityItemsAtPlant(self):
        for p in range(self.p):
            for i in range(self.i):
                for t in range(self.t):
                    self.model.add_constraint(
                        self.model.I_p_i_t[p, i, t] <= self.U_p_i[p][i],
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

    def createGoalProgrammingRestrictions(self):
        for t in range(self.t):
            self.model.add_constraints(
                [
                    self.f1[t] + self.negative[0, t] - self.positive[0, t]
                    == self.targets[t]["f1_target"],
                    self.f2[t] + self.negative[1, t] - self.positive[1, t]
                    == self.targets[t]["f2_target"],
                    self.f3[t] + self.negative[2, t] - self.positive[2, t]
                    == self.targets[t]["f3_target"],
                    self.f4[t] + self.negative[3, t] - self.positive[3, t]
                    == self.targets[t]["f4_target"],
                    self.f5[t] + self.negative[4, t] - self.positive[4, t]
                    == self.targets[t]["f5_target"],
                ]
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

        """self.log.info("*******************************")
        self.log.info("============ Z ================")
        self.log.info("*******************************")"""
        Z = []
        for t in range(self.t):
            # self.log.info(f"\n\n============ periodo {t} ============")
            v_list = []
            for v in range(self.v):
                # self.log.info(f"\n============ veiculo {v} ============")
                i_list = []
                for i in range(self.i):
                    k_list = []
                    for k in range(self.k):
                        # variable = abs(self.model.Z_v_i_k_t[v,i,k,t].solution_value)
                        variable = abs(
                            self.model.get_var_by_name(
                                f"Z_{v}_{i}_{k}_{t}"
                            ).solution_value
                        )
                        # self.log.info(f" origem: {i} destino: {k} == {variable}")
                        k_list.append(variable)
                    i_list.append(k_list)
                v_list.append(i_list)
            Z.append(v_list)
        # self.log.info("\n\n===============================\n\n")

        for t in range(len(Z)):
            # self.log.info(f"\n\n============ periodo {t} ============")
            v_list = []
            for v in range(len(Z[t])):
                # self.log.info(f"\n============ veiculo {v} ============")
                for i in range(len(Z[t][v])):
                    string = ""
                    for k in range(len(Z[t][v][i])):
                        string += str(Z[t][v][i][k]) + "\t"
                    # self.log.info(string)

        # self.log.info("*******************************")
        # self.log.info("============ Y ================")
        # self.log.info("*******************************")
        Y = []
        for t in range(self.t):
            # self.log.info(f"\n\n============ periodo {t} ============")
            p_list_y = []
            for p in range(self.p):
                variable = abs(self.model.get_var_by_name(f"Y_{p}_{t}").solution_value)
                p_list_y.append(variable)
                # print("produto: ",p," == ", variable)
            Y.append(p_list_y)
        # self.log.info("\n\n===============================\n\n")

        # self.log.info("*******************************")
        # self.log.info("============ X ================")
        # self.log.info("*******************************")
        X = []
        for t in range(self.t):
            # self.log.info(f"\n\n============ periodo {t} ============")
            p_list_x = []
            for p in range(self.p):
                variable = self.model.get_var_by_name(f"X_{p}_{t}").solution_value
                p_list_x.append(variable)
                # print("produto: ",p," == ", variable)
            X.append(p_list_x)
        # self.log.info("\n\n===============================\n\n")

        # self.log.info("*******************************")
        # self.log.info("============ I ================")
        # self.log.info("*******************************")
        I = []
        for t in range(self.t):
            # print("\n\n============ periodo ",t," ============")
            p_list_i = []
            for i in range(self.i):
                i_list_i = []
                # print("\n============ cliente ",i," ============")
                for p in range(self.p):
                    # print("produto: ",p," == ", self.I_p_i_t[p,i,t].x)
                    variable = self.model.get_var_by_name(
                        f"I_{p}_{i}_{t}"
                    ).solution_value
                    i_list_i.append(variable)
                p_list_i.append(i_list_i)
            I.append(p_list_i)
        # self.log.info("\n\n===============================\n\n")

        # self.log.info("*******************************")
        # self.log.info("============ R ================")
        # self.log.info("*******************************")
        R = []
        for t in range(self.t):
            # self.log.info(f"\n\n============ periodo {t} ============")
            t_list = []
            for v in range(self.v):
                # self.log.info(f"\n============ veiculo {v} ============")
                v_list = []
                for p in range(self.p):
                    p_list = []
                    for i in range(self.i):
                        i_list = []
                        for k in range(self.k):
                            # print("\n============ cliente ",i," -> cliente ",k," ============")
                            i_list.append(
                                float(
                                    self.model.get_var_by_name(
                                        f"R_{p}_{v}_{i}_{k}_{t}"
                                    ).solution_value
                                )
                            )
                            # self.log.info(f"\nperiodo {t} -> veiculo {v} -> cliente {i} -> cliente {k} -> produto {p} == { float(self.R_p_v_i_k_t[p,v,i,k,t].x)}")
                            # print("produto: ",p," == ", self.R_p_v_i_k_t[p,v,i,k,t].x)
                        p_list.append(i_list)
                    v_list.append(p_list)
                t_list.append(v_list)
            R.append(t_list)
        # self.log.info("\n\n===============================\n\n")

        # self.log.info("*******************************")
        # self.log.info("============ Q ================")
        # self.log.info("*******************************")
        Q = []
        for t in range(self.t):
            # self.log.info(f"\n\n============ periodo {t} ============")
            t_list = []
            for v in range(self.v):
                # self.log.info(f"\n============ veiculo {v} ============")
                v_list = []
                for p in range(self.p):
                    # self.log.info(f"\n============ cliente {i} ============")
                    p_list = []
                    for i in range(self.i):
                        # self.log.info(f"produto:{p} == {self.Q_p_v_i_t[p,v,i,t].x}")
                        variable = self.model.get_var_by_name(
                            f"Q_{p}_{v}_{i}_{t}"
                        ).solution_value
                        p_list.append(variable)
                    v_list.append(p_list)
                t_list.append(v_list)
            Q.append(t_list)
        # self.log.info("\n\n===============================\n\n")

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
            self.log.info("Relaxation model replaced")
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

    def processInformationsSolver(self):
        if self.model.solve_details:
            self.solCount = 1 if self.model.solution else 0
            self.objBound = (
                self.model.solve_details.best_bound
                if hasattr(self.model.solve_details, "best_bound")
                else 0
            )
            self.nodeCount = (
                self.model.solve_details.nb_nodes_processed
                if hasattr(self.model.solve_details, "nb_nodes_processed")
                else 0
            )
        else:
            self.solCount = 0
            self.objBound = 0
            self.nodeCount = 0

    def _record_mip_telemetry(self, elapsed_seconds, has_incumbent, relative_gap):
        if not self.telemetry_config["enabled"]:
            return
        if has_incumbent and self._first_feasible_seconds is None:
            self._first_feasible_seconds = float(elapsed_seconds)
            self._telemetry_events.append({"event": "first_feasible", "elapsed_seconds": float(elapsed_seconds), "source": "mip"})
        if relative_gap is not None and relative_gap <= self.telemetry_config["gap_target_relative"] and self._gap_target_seconds is None:
            self._gap_target_seconds = float(elapsed_seconds)
            self._telemetry_events.append({"event": "gap_target", "elapsed_seconds": float(elapsed_seconds), "gap": float(relative_gap)})

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
        return {
            "strategy": "solver",
            "mip_seconds": float(self.time), "status": status, "timed_out": timed_out,
            "objective": objective, "best_bound": getattr(details, "best_bound", None),
            "relative_gap": gap, "node_count": self.nodeCount, "solution_count": self.solCount,
            "first_feasible_seconds": self._first_feasible_seconds,
            "gap_target_seconds": self._gap_target_seconds,
            "mip_events": list(self._telemetry_events), "pso_iterations": [],
        }

    def _install_mip_telemetry_callback(self):
        if not self.telemetry_config["enabled"]:
            return
        try:
            from cplex.callbacks import MIPInfoCallback
            owner = self
            started_at = time.perf_counter()
            class TelemetryCallback(MIPInfoCallback):
                def __call__(self):
                    try:
                        elapsed = time.perf_counter() - started_at
                        incumbent = self.has_incumbent()
                        gap = self.get_MIP_relative_gap() if incumbent else None
                        owner._record_mip_telemetry(elapsed, incumbent, gap)
                    except Exception:
                        pass
            self.model.register_callback(TelemetryCallback)
        except Exception as error:
            self.log.warning(f"Telemetria MIP sem callback: {error}")

    def solver(self, numThreads=None, timeLimit=None):
        self.createDecisionVariables()

        self.log.info(f"Variabes.start == {self.start['start']}")
        if self.start["start"] == True:
            self.startVariables()

        self.crateObjectiveFunction()
        self.log.info("Objetivo criado")
        if Config.get_nested("solver", "multiobjective"):
            self.createGoalProgrammingRestrictions()
            self.createEpsilonRestricted()
        self.createEstablishInvetoryBalanceAtPlant()
        self.log.info("Balanceamento estoque Planta criado")
        self.creteInventoryBalancingInventoryCustomers()
        self.log.info("Balanceamento estoque Cliente criado")
        self.createPlantsMaximum()
        self.log.info("Planta max criado")
        self.createRelationshipBetweenProduction()
        self.log.info("Produção criado")
        self.createDelimitMaximumCapacityItemsAtPlant()
        self.log.info("Capacidade maxima planta criado")
        self.createVehiclePreventTransshipmentIntermediateNodes()
        self.log.info("Transshipment criado")
        self.createEliminationSubroutes()
        self.log.info("Subrotas criado")
        self.createVehicleLoadCapacityDelimited()
        self.log.info("Capacidade maxima veículo criado")
        self.createImposeMostOneRouteEachVehicle()
        self.log.info("Max rota veículo criado")
        self.createEnsureRoutesOnlyPlant()
        self.log.info("Rota somente entre plantas criado")
        self.createVehicleMostVisitCustomerEachPeriod()
        self.log.info("Veículo visita cliente criado")
        # self.outModel()
        if Config.get_nested("postprocessing", "build_target"):
            self.generteRelax(
                REPLACE_MODEL=Config.get_nested("relaxed_solution", "replace_model")
            )
            self.log.info("Solução relaxada gerada")
        else:
            self.log.info("Solução não relaxada - usando modelo original")

        # Set parameters
        if timeLimit is not None:
            self.model.set_time_limit(timeLimit)
        # if numThreads is not None:
        # self.model.context.cplex_parameters.threads = numThreads

        start_time = time.time()
        self._telemetry_started_at = time.perf_counter()
        self._install_mip_telemetry_callback()
        if not Config.get_nested("relaxed_solution", "use"):
            self.solution = self.model.solve(log_output=False)
        end_time = time.time()
        self.time = end_time - start_time

        self.processInformationsSolver()
        elapsed = time.perf_counter() - self._telemetry_started_at
        details = getattr(self.model, "solve_details", None)
        self._record_mip_telemetry(elapsed, self.solCount > 0, getattr(details, "mip_relative_gap", None))
