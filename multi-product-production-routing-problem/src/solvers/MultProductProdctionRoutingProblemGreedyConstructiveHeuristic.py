#################################################################################################
# Multi Product Prodction Routing Problem Greedy Constructive Heuristic
# Copyright 2024 Mateus Chacon

# Este programa é um software livre, você pode redistribuí-lo e/ou modificá-lo
# sob os termos da Licença Pública Geral GNU como publicada pela Fundação do Software Livre (FSF),
# na versão 3 da Licença, ou (a seu critério) qualquer versão posterior.

# Este programa é distribuído na esperança de que possa ser útil, mas SEM NENHUMA GARANTIA,
# e sem uma garantia implícita de ADEQUAÇÃO a qualquer MERCADO ou APLICAÇÃO EM PARTICULAR.

# Veja a Licença Pública Geral GNU para mais detalhes
#################################################################################################
from src.log.Logger import Logger
from typing import List
from src.solvers.GreedyRandomizedConstructionRoute import GreedyRandomizedConstructionRoute as GR
from src.helpers.ReadPrpFile import ReadPrpFile as RD
from src.solvers.MultProductProdctionRoutingProblem import MultProductProdctionRoutingProblem as MPPRP
import pdb
import numpy as np
import math
import json
import time

class MultProductProdctionRoutingProblemGreedyConstructiveHeuristic:

    def __init__(self,map,dir,log:Logger,rng:np.random.Generator):
        self.data = map
        self.p=map['num_products']            ##Products  
        self.i=map['num_customers'] + 1       ##Customers
        self.k=map['num_customers'] + 1       ##Customers
        self.t=map['num_periods']             ##Periods
        self.v=map['num_vehicles']            ##Vehicles         
        self.B=map['B']                       ##Production capacity;
        self.b_p=map['b_p']                   ##Time required to produce item 𝑝;
        self.c_p=map['c_p']                   ##Production cost of item 𝑝;
        self.s_p=map['s_p']                   ##Setup cost of item 𝑝;
        self.M=map['M']                       ##Big number 
        self.U_p_i=map['U_pi']                ##Maximum inventory upper bound of item 𝑝 at site i;
        self.I_p_i_0=map['I_pi0']             ##Initial Inventory of item 𝑝 at site 𝑖;
        self.h_p_i=map['h_pi']                ##Inventory cost of item 𝑝 at site 𝑖;
        self.C=map['C']                       ##Vehicle capacity;
        self.f=map['f']                       ##Fixed transportation cost;
        self.a_i_k=map['a_ik']                ##Transportation cost for traveling from node 𝑖 to node k;
        self.d_p_i_t=map['d_pit']             ##Demand of item 𝑝 at customer 𝑖 in period 𝑡.
        self.X_p_t={}                         ##Quantity of item 𝑝 produced in period 𝑡.
        self.Y_p_t={}                         ##1, if item 𝑝 is produced in period 𝑡; or 0, otherwise.
        self.I_p_i_t={}                       ##Inventory of item 𝑝 at site 𝑖 in the end of period 𝑡.
        self.Z_v_i_k_t={}                     ##1, if vehicle v travels along edge (i,k) in period t; or 0, atherwise.
        self.R_p_v_i_k_t={}                   ##Quantity of item 𝑝 transported by vehicle 𝑣 on edge (𝑖, 𝑘) in period 𝑡;
        self.Q_p_v_i_t={}                     ##Quantity of item 𝑝 delivered by vehicle 𝑣 to customer 𝑖 in period 𝑡.
        self.dir = dir
        self.time = 0
        self.solCount = 0
        self.log:Logger = log
        self.max_inter = 100
        self.alfa = 0.2
        self.seed = 123
        self.greedyRoute = GR(log=log)
        self.variables={}
        self.solution={}
        self.mitStart = False
        self.solverGurobi = 0
        self.rng = rng
        self.feasibility_report = {"feasible": False, "violations": []}

    def setMitStart(self,mitStart):
        self.mitStart = mitStart

    def setMaxInter(self,max_inter):
        self.max_inter = max_inter
    
    def setAlfa(self,alfa):
        self.alfa = alfa
    
    def setSeed(self,seed):
        self.seed = seed

    def getDistancesInPeriod(self,candidates):
        D = np.zeros((len(candidates), len(candidates)), dtype=float)
        for i in range(len(candidates)):
            for k in range(len(candidates)):
                d = self.a_i_k[candidates[i]][candidates[k]]
                D[i, k] = d
                D[k, i] = d
        return D


    def construirSolucao(self):
        solucao_t_i_p = [[[{'cliente': i,'produto': p,'periodo': t,'estoque': 0,'demanda': 0,'producaco': 0} for p in range(self.p)] for i in range(self.i)] for t in range(self.t)]
        estoque_i_p = [[self.I_p_i_0[p][i] for p in range(self.p)] for i in range(self.i)]
        capacities = [[self.C] for _ in range(self.v)]
        demands_t = []
        dist_t = []
        routes = []
        points_t = []
        candidates_soluction = []

        for t in range(self.t):
            capacidade_producao_restante = int(self.B)
            capacidade_veiculo_totais = [self.C for _ in range(self.v)]

            # pdb.set_trace()
            candidatos = []
            veiculo_corrente=0
            for i in range(1, self.i):
                for p in range(self.p):
                    demanda_t = self.d_p_i_t[p][i-1][t]
                    qte = 0

                    if(estoque_i_p[i][p] < demanda_t): 
                        produto_faltante = demanda_t - estoque_i_p[i][p]
                        disponibilidade_estoque = self.U_p_i[p][i] - estoque_i_p[i][p]
                        while veiculo_corrente < self.v and capacidade_veiculo_totais[veiculo_corrente] <= 0:
                            veiculo_corrente += 1

                        if veiculo_corrente >= self.v:
                            self.log.error("modelo Inviavel pelas restrições de capacidade do veiculo")
                            qte = 0
                        else:
                            qte = min(
                                produto_faltante,
                                capacidade_producao_restante,
                                disponibilidade_estoque,
                                capacidade_veiculo_totais[veiculo_corrente],
                            )

                            capacidade_producao_restante -=qte
                            capacidade_veiculo_totais[veiculo_corrente] -= qte

                            if(capacidade_veiculo_totais[veiculo_corrente] < 0):

                                capacidade_veiculo_totais[veiculo_corrente] += qte

                                veiculo_corrente+=1
                                if(veiculo_corrente>=self.v):
                                    self.log.error("modelo Inviavel pelas restrições de capacidade do veiculo")
                                else:
                                    capacidade_veiculo_totais[veiculo_corrente] -= qte

                            if(capacidade_producao_restante<0):
                                self.log.error("modelo Inviavel pelas restrições de produção")
                                break

                    assert qte >= 0

                    estoque_i_p[i][p] = max(0, estoque_i_p[i][p] + qte - demanda_t)
                    solucao_t_i_p[t][i][p]['producaco']+= qte
                    solucao_t_i_p[t][i][p]['cliente'] = i
                    solucao_t_i_p[t][i][p]['produto'] = p
                    solucao_t_i_p[t][i][p]['periodo'] = t
                    solucao_t_i_p[t][i][p]['estoque'] = estoque_i_p[i][p]
                    solucao_t_i_p[t][i][p]['demanda'] = demanda_t
                        
                    if(
                        estoque_i_p[i][p] < self.U_p_i[p][i]
                        and capacidade_producao_restante > 0
                        and veiculo_corrente < self.v
                        and capacidade_veiculo_totais[veiculo_corrente] > 0
                    ):
                        demanda_futura = sum(self.d_p_i_t[p][i-1][t+1:self.t])
                        custo_unitario = self.s_p[p] + self.c_p[p] + self.h_p_i[p][i]*demanda_futura
                        candidatos.append((i,p,'barato',1/custo_unitario))

            candidatos.sort(key=lambda x: x[3], reverse=True)

            iter=0
            positivos = [v for v in capacidade_veiculo_totais if v > 0]
            if positivos:
                maior_valor = max(positivos)
                veiculo_corrente = capacidade_veiculo_totais.index(maior_valor)

            while (
                capacidade_producao_restante > 0
                and len(candidatos) != 0
                and veiculo_corrente < self.v
                and capacidade_veiculo_totais[veiculo_corrente] > 0
                and iter <= len(candidatos)
            ):
                top_k = math.ceil(self.alfa * len(candidatos))
                RCL = candidatos[:top_k]
                i,_,_,_ = self.rng.choice(RCL)
                i = int(i)
                total_p = []
                for p in range(self.p):
                    demanda_futura = sum(self.d_p_i_t[p][i-1][t+1:self.t])
                    disponibilidade_estoque = self.U_p_i[p][i] - estoque_i_p[i][p]

                    faltante = demanda_futura - estoque_i_p[i][p]
                    if(faltante<0):
                        faltante = 0

                    total_p.append(
                        min(
                            capacidade_producao_restante,
                            demanda_futura,
                            disponibilidade_estoque,
                            faltante,
                            capacidade_veiculo_totais[veiculo_corrente],
                        )
                    )


                capacidade_producao_restante -= sum(total_p)
                capacidade_veiculo_totais[veiculo_corrente] -= sum(total_p)

                if(capacidade_producao_restante<0 or capacidade_veiculo_totais[veiculo_corrente]< 0 ):
                    self.log.warning(f"Produção está negátiva: {capacidade_producao_restante} ou capcidade_veiculo negativo:{capacidade_veiculo_totais[veiculo_corrente]}")
                    capacidade_producao_restante += sum(total_p)
                    capacidade_veiculo_totais[veiculo_corrente] += sum(total_p)
                    iter+=1
                    positivos = [v for v in capacidade_veiculo_totais if v > 0]
                    if positivos:
                        maior_valor = max(positivos)
                        veiculo_corrente = capacidade_veiculo_totais.index(maior_valor)
                    
                else:
                    for p in range(len(total_p)):
                        demanda_atual = self.d_p_i_t[p][i-1][t]
                        estoque_i_p[i][p] = max(0, estoque_i_p[i][p] + total_p[p] - demanda_atual)
                        solucao_t_i_p[t][i][p]['producaco']+= total_p[p]
                        solucao_t_i_p[t][i][p]['cliente'] = i
                        solucao_t_i_p[t][i][p]['produto'] = p
                        solucao_t_i_p[t][i][p]['periodo'] = t
                        solucao_t_i_p[t][i][p]['estoque'] = estoque_i_p[i][p]
                        solucao_t_i_p[t][i][p]['demanda'] = demanda_atual
                        candidatos = [c for c in candidatos if not (c[0]==i and c[1]==p)]

            candidates_t = [0]  # 0 = depósito
            dem_t = [[0.0] * self.p]  # vetor de produtos no depósito (zerado)

            for i, linha in enumerate(solucao_t_i_p[t]):
                prod = []
                client_current = 0
                for p, celula in enumerate(linha):
                    prod.append(celula['producaco'])  # já é separado por produto
                    estoque_i_p[i][p] = celula['estoque']
                    client_current = celula['cliente']

                if any(q > 0 for q in prod):  # se tem algo produzido para o cliente
                    candidates_t.append(client_current)
                    dem_t.append(prod)  # agora vai com vetor por produto, não o somatório

            # pdb.set_trace()
            candidates_t = list(dict.fromkeys(candidates_t))
            dem_t = [v for v in dem_t if v]
            D = self.getDistancesInPeriod(candidates_t)
           
            route_result = self.greedyRoute.greedyRandomizedConstruction(
                candidates_t,
                dem_t,
                capacities,
                D,
                int(self.v),
                self.alfa,
                self.rng,
            )
            if route_result is None:
                self.log.error(
                    f"heuristica de rota falhou no periodo {t}; registrando rota vazia"
                )
                route = [[] for _ in range(self.v)]
                distance = 0
                demandas = [{"veiculo": v, "entregas": []} for v in range(self.v)]
                points = [[] for _ in range(self.v)]
            else:
                route, distance, demandas, points = route_result
            routes.append({'periodo':t ,'route':route,'distance':distance,'demandas':demandas})
            
            demands_t.append(dem_t)
            dist_t.append(D)
            points_t.append(points)
            candidates_soluction.append(candidates_t)

        self.solution= {
            "production": solucao_t_i_p,
            "routes": routes,
            "candidates": candidates_soluction,
            "demands": demands_t,
            "distancies": dist_t,
            "points": points_t
        }
        #self.log.info(json.dumps(final_solution, indent=4))

    def convertVariables(self):

        final_solution = self.solution
        Z = np.zeros((self.v,self.i,self.k,self.t), dtype=int)

        '''𝑧𝑣𝑖𝑘𝑡'''
        for t in range(len(final_solution["routes"])):
            for v in range(len(final_solution["routes"][t]["route"])):
                for i in range( len(final_solution["routes"][t]["route"][v])):
                    origem = final_solution["routes"][t]["route"][v][i]
                    if(i+1 == len(final_solution["routes"][t]["route"][v])):
                        destino = 0
                    else:
                        destino = final_solution["routes"][t]["route"][v][i+1]
                    Z[v, origem,destino, t] = 1
        R = np.zeros((self.p,self.v,self.i,self.k,self.t), dtype=int)
        Q = np.zeros((self.p,self.v,self.i,self.t), dtype=int)

        '''𝑟𝑝𝑣𝑖𝑘𝑡'''
        '''𝑞𝑝𝑣𝑖𝑡'''
        for t in range(len(final_solution["routes"])):
            for v in range(len(final_solution["routes"][t]["demandas"])):
                for i in range( len(final_solution["routes"][t]["demandas"][v]['entregas'])):
                    origem_i = final_solution["routes"][t]["demandas"][v]["entregas"][i]["cliente"]
                    if(i+1 == len(final_solution["routes"][t]["route"][v])):
                        destino_j = 0
                    else:
                        destino_j = final_solution["routes"][t]["demandas"][v]["entregas"][i+1]["cliente"]
                    for p in range(len(final_solution["routes"][t]["demandas"][v]["entregas"][i]["produtos"])):
                        R[p,v,origem_i,destino_j,t] = final_solution["routes"][t]["demandas"][v]["entregas"][i]["produtos"][p]["restante_veiculo"]
                for i in range( len(final_solution["routes"][t]["demandas"][v]['entregas'])):
                    origem_i = final_solution["routes"][t]["demandas"][v]["entregas"][i]["cliente"]
                    for p in range(len(final_solution["routes"][t]["demandas"][v]["entregas"][i]["produtos"])):
                        Q[p,v,origem_i,t] = final_solution["routes"][t]["demandas"][v]["entregas"][i]["produtos"][p]["qte_entregue"]

        X = np.zeros((self.p,self.t), dtype=int)
        Y = np.zeros((self.p,self.t), dtype=int)
        I = np.zeros((self.p,self.i,self.t), dtype=int)
        '''𝐼𝑝𝑖𝑡'''
        for t in range(len(final_solution["production"])):
            producao = np.zeros((self.p), dtype=int)
            for i in range(len(final_solution["production"][t])):
                for p in range(len(final_solution["production"][t][i])):
                    producao[p]+=final_solution["production"][t][i][p]["producaco"]
                    I[p,i,t] = final_solution["production"][t][i][p]["estoque"]


            for p in range(len(final_solution["production"][t][i])):
                if(producao[p]>0):
                    Y[p,t] = 1
                X[p,t] = producao[p]

        self.variables={"X":X, "Y":Y, "I":I, "Q":Q, "R":R, "Z":Z}

        return Z,X,Y,I,R,Q

    def _add_feasibility_violation(self, violations, message, max_violations=50):
        if len(violations) < max_violations:
            violations.append(message)

    def validateFeasibility(self, eps=1e-6):
        violations = []
        X = self.variables["X"]
        Y = self.variables["Y"]
        I = self.variables["I"]
        Q = self.variables["Q"]
        R = self.variables["R"]
        Z = self.variables["Z"]

        for p in range(self.p):
            for t in range(self.t):
                if X[p, t] < -eps:
                    self._add_feasibility_violation(violations, f"X[{p},{t}] negativo")
                if Y[p, t] not in (0, 1):
                    self._add_feasibility_violation(violations, f"Y[{p},{t}] nao binario")
                if X[p, t] - self.M * Y[p, t] > eps:
                    self._add_feasibility_violation(violations, f"X[{p},{t}] excede M*Y")

        for t in range(self.t):
            total_production_time = sum(self.b_p[p] * X[p, t] for p in range(self.p))
            if total_production_time - self.B > eps:
                self._add_feasibility_violation(
                    violations,
                    f"capacidade de producao excedida no periodo {t}: {total_production_time}>{self.B}",
                )

        for p in range(self.p):
            for i in range(self.i):
                for t in range(self.t):
                    if I[p, i, t] < -eps:
                        self._add_feasibility_violation(violations, f"I[{p},{i},{t}] negativo")
                    if I[p, i, t] - self.U_p_i[p][i] > eps:
                        self._add_feasibility_violation(violations, f"I[{p},{i},{t}] excede U")

        for p in range(self.p):
            for t in range(self.t):
                delivered_from_plant = sum(Q[p, v, i, t] for v in range(self.v) for i in range(1, self.i))
                previous_plant_inventory = self.I_p_i_0[p][0] if t == 0 else I[p, 0, t - 1]
                expected_plant_inventory = previous_plant_inventory + X[p, t] - delivered_from_plant
                if abs(expected_plant_inventory - I[p, 0, t]) > eps:
                    self._add_feasibility_violation(
                        violations,
                        f"balanco da planta violado p={p} t={t}",
                    )

        for p in range(self.p):
            for i in range(1, self.i):
                for t in range(self.t):
                    delivered_to_customer = sum(Q[p, v, i, t] for v in range(self.v))
                    previous_customer_inventory = self.I_p_i_0[p][i] if t == 0 else I[p, i, t - 1]
                    expected_customer_inventory = (
                        previous_customer_inventory
                        + delivered_to_customer
                        - self.d_p_i_t[p][i - 1][t]
                    )
                    if abs(expected_customer_inventory - I[p, i, t]) > eps:
                        self._add_feasibility_violation(
                            violations,
                            f"balanco do cliente violado p={p} i={i} t={t}",
                        )

        for v in range(self.v):
            for i in range(self.i):
                for k in range(self.k):
                    for t in range(self.t):
                        if i == k:
                            continue
                        load = sum(R[p, v, i, k, t] for p in range(self.p))
                        if load - self.C * Z[v, i, k, t] > eps:
                            self._add_feasibility_violation(
                                violations,
                                f"capacidade de veiculo violada v={v} i={i} k={k} t={t}",
                            )

        for p in range(self.p):
            for v in range(self.v):
                for k in range(1, self.k):
                    for t in range(self.t):
                        incoming = sum(R[p, v, i, k, t] for i in range(self.i) if i != k)
                        outgoing = sum(R[p, v, k, l, t] for l in range(self.i) if l != k)
                        if abs(incoming - outgoing - Q[p, v, k, t]) > eps:
                            self._add_feasibility_violation(
                                violations,
                                f"fluxo de carga violado p={p} v={v} k={k} t={t}",
                            )

        for v in range(self.v):
            for t in range(self.t):
                plant_departures = sum(Z[v, 0, k, t] for k in range(1, self.k))
                if plant_departures - 1 > eps:
                    self._add_feasibility_violation(
                        violations,
                        f"mais de uma rota do veiculo {v} no periodo {t}",
                    )

        for k in range(1, self.k):
            for t in range(self.t):
                visits = sum(Z[v, i, k, t] for v in range(self.v) for i in range(self.i) if i != k)
                if visits - 1 > eps:
                    self._add_feasibility_violation(
                        violations,
                        f"cliente {k} visitado mais de uma vez no periodo {t}",
                    )

        for v in range(self.v):
            for k in range(self.k):
                for t in range(self.t):
                    incoming = sum(Z[v, i, k, t] for i in range(self.i) if i != k)
                    outgoing = sum(Z[v, k, l, t] for l in range(self.i) if l != k)
                    if abs(incoming - outgoing) > eps:
                        self._add_feasibility_violation(
                            violations,
                            f"fluxo de rota violado v={v} k={k} t={t}",
                        )

        return {"feasible": len(violations) == 0, "violations": violations}

    def _customers_priority_from_current_solution(self, t):
        priority = []
        seen = set()

        for route_info in self.solution.get("routes", []):
            if route_info.get("periodo") != t:
                continue
            for route in route_info.get("route", []):
                for client in route:
                    if client != 0 and client not in seen:
                        priority.append(client)
                        seen.add(client)

        if "Q" in self.variables:
            Q = self.variables["Q"]
            delivered = []
            for i in range(1, self.i):
                amount = sum(Q[p, v, i, t] for p in range(self.p) for v in range(self.v))
                if amount > 0 and i not in seen:
                    delivered.append((i, amount))
            delivered.sort(key=lambda item: item[1], reverse=True)
            for i, _ in delivered:
                priority.append(i)
                seen.add(i)

        deficits = []
        for i in range(1, self.i):
            total_demand = sum(self.d_p_i_t[p][i - 1][t] for p in range(self.p))
            if i not in seen:
                deficits.append((i, total_demand))
        deficits.sort(key=lambda item: item[1], reverse=True)
        priority.extend(i for i, _ in deficits)
        return priority

    def _assign_deliveries_to_vehicles(self, deliveries_by_client):
        Q_t = np.zeros((self.p, self.v, self.i), dtype=int)
        routes_t = [[] for _ in range(self.v)]
        vehicle_loads = [0 for _ in range(self.v)]

        for client, delivery in deliveries_by_client:
            total_delivery = sum(delivery)
            if total_delivery == 0:
                continue

            assigned = False
            for v in range(self.v):
                if vehicle_loads[v] + total_delivery <= self.C:
                    routes_t[v].append(client)
                    vehicle_loads[v] += total_delivery
                    for p in range(self.p):
                        Q_t[p, v, client] = delivery[p]
                    assigned = True
                    break

            if not assigned:
                return None, None

        return Q_t, routes_t

    def _build_route_variables_from_routes(self, Q, routes_by_period):
        Z = np.zeros((self.v, self.i, self.k, self.t), dtype=int)
        R = np.zeros((self.p, self.v, self.i, self.k, self.t), dtype=int)

        for t in range(self.t):
            for v in range(self.v):
                route = routes_by_period[t][v]
                if not route:
                    continue

                full_route = [0] + route + [0]
                for pos in range(len(full_route) - 1):
                    origin = full_route[pos]
                    destination = full_route[pos + 1]
                    Z[v, origin, destination, t] = 1

                    remaining_clients = full_route[pos + 1 : -1]
                    for p in range(self.p):
                        R[p, v, origin, destination, t] = sum(
                            Q[p, v, client, t] for client in remaining_clients
                        )

        return Z, R

    def repairSolution(self):
        X = np.zeros((self.p, self.t), dtype=int)
        Y = np.zeros((self.p, self.t), dtype=int)
        I = np.zeros((self.p, self.i, self.t), dtype=int)
        Q = np.zeros((self.p, self.v, self.i, self.t), dtype=int)
        routes_by_period = [[[] for _ in range(self.v)] for _ in range(self.t)]

        previous_inventory = np.array(self.I_p_i_0, dtype=int)

        for t in range(self.t):
            deliveries_by_client = []
            production_needed = [0 for _ in range(self.p)]

            for i in self._customers_priority_from_current_solution(t):
                delivery = []
                for p in range(self.p):
                    demand = self.d_p_i_t[p][i - 1][t]
                    deficit = max(0, demand - previous_inventory[p][i])
                    available_storage = self.U_p_i[p][i] - previous_inventory[p][i]
                    delivery.append(int(max(0, min(deficit, available_storage))))
                deliveries_by_client.append((i, delivery))

            total_delivery_by_product = [
                sum(delivery[p] for _, delivery in deliveries_by_client)
                for p in range(self.p)
            ]

            for p in range(self.p):
                production_needed[p] = max(
                    0, total_delivery_by_product[p] - previous_inventory[p][0]
                )

            production_time = sum(
                self.b_p[p] * production_needed[p] for p in range(self.p)
            )
            if production_time > self.B:
                self.log.warning(
                    f"reparo inviavel no periodo {t}: capacidade de producao insuficiente"
                )
                self.variables = {"X": X, "Y": Y, "I": I, "Q": Q, "R": np.zeros((self.p,self.v,self.i,self.k,self.t), dtype=int), "Z": np.zeros((self.v,self.i,self.k,self.t), dtype=int)}
                return False

            Q_t, routes_t = self._assign_deliveries_to_vehicles(deliveries_by_client)
            if Q_t is None:
                self.log.warning(
                    f"reparo inviavel no periodo {t}: capacidade de veiculo insuficiente"
                )
                self.variables = {"X": X, "Y": Y, "I": I, "Q": Q, "R": np.zeros((self.p,self.v,self.i,self.k,self.t), dtype=int), "Z": np.zeros((self.v,self.i,self.k,self.t), dtype=int)}
                return False

            routes_by_period[t] = routes_t
            for p in range(self.p):
                X[p, t] = production_needed[p]
                Y[p, t] = 1 if X[p, t] > 0 else 0
                for v in range(self.v):
                    for i in range(self.i):
                        Q[p, v, i, t] = Q_t[p, v, i]

                delivered_from_plant = sum(Q[p, v, i, t] for v in range(self.v) for i in range(1, self.i))
                I[p, 0, t] = previous_inventory[p][0] + X[p, t] - delivered_from_plant

                for i in range(1, self.i):
                    delivered_to_customer = sum(Q[p, v, i, t] for v in range(self.v))
                    I[p, i, t] = (
                        previous_inventory[p][i]
                        + delivered_to_customer
                        - self.d_p_i_t[p][i - 1][t]
                    )

            previous_inventory = I[:, :, t].copy()

        Z, R = self._build_route_variables_from_routes(Q, routes_by_period)
        self.variables = {"X": X, "Y": Y, "I": I, "Q": Q, "R": R, "Z": Z}
        return True
    
    def getResultsSolverHeurisct(self):
        z=self.variables["Z"]
        x=self.variables["X"]
        y=self.variables["Y"]
        ii=self.variables["I"]
        r=self.variables["R"]
        q=self.variables["Q"]
        '''print("*******************************")
        print("============ Z ================")
        print("*******************************")'''
        Z=[]
        for t in range(self.t):
            #print("\n\n============ periodo ",t," ============")
            v_list =[]
            for v in range(self.v):
                #print("\n============ veiculo ",v," ============")
                i_list =[]
                for i in range(self.i):
                    k_list=[]
                    for k in range(self.k):
                        variable = z[v,i,k,t]
                        #print(" origem: ",i," destino: ",k," == ",variable)
                        k_list.append(int(variable))
                    i_list.append(k_list)
                v_list.append(i_list)
            Z.append(v_list)
        #print("\n\n===============================\n\n")
        '''for t in range(len(Z)):
            print("\n\n============ periodo ",t," ============")
            for v in range(len(Z[t])):
                print("\n============ veiculo ",v," ============")
                for i in range(len(Z[t][v])):
                    string = ""
                    for k in range(len(Z[t][v][i])):
                        string+= str(Z[t][v][i][k]) + "\t"
                    print(string)'''

        '''print("*******************************")
        print("============ Y ================")
        print("*******************************")'''
        Y = []
        for t in range(self.t):
            #print("\n\n============ periodo ",t," ============")
            p_list_y=[]
            for p in range(self.p):
                variable = abs(y[p,t])
                p_list_y.append(int(variable))
                #print("produto: ",p," == ", variable)
            Y.append(p_list_y)
        #print("\n\n===============================\n\n")
        '''print("*******************************")
        print("============ X ================")
        print("*******************************")'''
        X = []
        for t in range(self.t):
            #print("\n\n============ periodo ",t," ============")
            p_list_x=[]
            for p in range(self.p):
                p_list_x.append(int(x[p,t]))
                #print("produto: ",p," == ",x[p,t])
            X.append(p_list_x)
        '''print("\n\n===============================\n\n")
        print("*******************************")
        print("============ I ================")
        print("*******************************")'''
        I=[]
        for t in range(self.t):
            #print("\n\n============ periodo ",t," ============")
            p_list_i=[]
            for i in range(self.i):
                i_list_i=[]
                #print("\n============ cliente ",i," ============")
                for p in range(self.p):
                    #print("produto: ",p," == ", ii[p,i,t])
                    i_list_i.append(int(ii[p,i,t]))
                p_list_i.append(i_list_i)
            I.append(p_list_i)
        '''print("\n\n===============================\n\n")
        print("*******************************")
        print("============ R ================")
        print("*******************************")'''
        R=[]
        for t in range(self.t):
            #print("\n\n============ periodo ",t," ============")
            t_list=[]
            for v in range(self.v):
                #print("\n============ veiculo ",v," ============")
                v_list=[]
                for p in range(self.p):
                    p_list=[]
                    for i in range(self.i):
                        i_list=[]
                        for k in range(self.k):
                            #self.log.info(f"\nperiodo {t} -> veiculo {v} -> cliente {i} -> cliente {k} -> produto {p} == { r[p,v,i,k,t]}",)
                            #print("\n============ cliente ",i," -> cliente ",k," ============")
                            i_list.append(float(r[p,v,i,k,t]))
                            #print("produto: ",p," == ", r[p,v,i,k,t])
                        p_list.append(i_list)
                    v_list.append(p_list)
                t_list.append(v_list)
            R.append(t_list)
        '''print("\n\n===============================\n\n")
        print("*******************************")
        print("============ Q ================")
        print("*******************************")'''
        Q=[]
        for t in range(self.t):
            #print("\n\n============ periodo ",t," ============")
            t_list=[]
            for v in range(self.v):
                #print("\n============ veiculo ",v," ============")
                v_list=[]
                for p in range(self.p):
                    #print("\n============ cliente ",i," ============")
                    p_list=[]
                    for i in range(self.i):
                        #self.log.info(f"\nperiodo {t} -> veiculo {v} -> cliente -> {i} -> produto {p} == { q[p,v,i,t]}",)
                        #print("produto: ",p," == ",q[p,v,i,t])
                        p_list.append(int(q[p,v,i,t]))
                    v_list.append(p_list)
                t_list.append(v_list)
            Q.append(t_list)
        #print("\n\n===============================\n\n")
    
        FO = self.getValueObjectiveFunction()
        GAP = 0.0
        TIME = self.time
        EPSILON = None
        SOL_COUNT = self.solCount
        RELAXED_MODEL_OBJE_VAL = 0
        NODE_COUNT = 0
        OBJ_BOUND = FO
        NEW_TARGETS = None

        return (
            Z,
            X,
            Y,
            I,
            R,
            Q,
            [],
            FO,
            GAP,
            TIME,
            EPSILON,
            SOL_COUNT,
            RELAXED_MODEL_OBJE_VAL,
            NODE_COUNT,
            OBJ_BOUND,
            NEW_TARGETS,
        )

    def solver(self,numThreads=None,timeLimit=None):
        started_at = time.time()
        self.construirSolucao()
        self.convertVariables()
        self.feasibility_report = self.validateFeasibility()
        if not self.feasibility_report["feasible"]:
            original_variables = self.variables
            self.log.warning(
                "tentando reparar solucao heuristica inviavel: "
                + "; ".join(self.feasibility_report["violations"][:5])
            )
            if self.repairSolution():
                self.feasibility_report = self.validateFeasibility()
            else:
                self.variables = original_variables
                self.feasibility_report = self.validateFeasibility()
        self.time = time.time() - started_at
        self.solCount = 1
        if(self.mitStart==True):
            if self.feasibility_report["feasible"]:
                self.solverGurobi = MPPRP(self.data,self.dir,self.log,{"start":True, "variables":self.variables})
            else:
                self.log.warning(
                    "warm start da heuristica ignorado por inviabilidade: "
                    + "; ".join(self.feasibility_report["violations"][:5])
                )
                self.solverGurobi = MPPRP(self.data,self.dir,self.log,{"start":False})
            self.solverGurobi.solver(timeLimit=timeLimit,numThreads=numThreads)
           
    
    def getResults(self):
        if(self.mitStart==True):
            return self.solverGurobi.getResults()

        return self.getResultsSolverHeurisct()

    def getSolution(self):
        return self.solution

    def getValueObjectiveFunction(self):
        objExpr_1 = 0
        for p in range(self.p):
            for t in range(self.t):
                objExpr_1 += self.s_p[p] * self.variables["Y"][p][t] + self.c_p[p] * self.variables["X"][p][t]

        objExpr_2 = 0
        for p in range(self.p):
            for i in range(self.i):
                for t in range(self.t):
                    objExpr_2+=self.h_p_i[p][i]*self.variables["I"][p][i][t]

        objExpr_3 = 0
        for v in range(self.v):
            for k in range(1,self.k):
                for t in range(self.t):
                    objExpr_3+=self.f*self.variables["Z"][v][0][k][t]

        objExpr_4 = 0
        for v in range(self.v):
            for i in range(self.i):
                for k in range(self.k):
                    if(i!=k):
                        for t in range(self.t):
                            objExpr_4+=self.a_i_k[i][k]*self.variables["Z"][v][i][k][t]

        return objExpr_1 + objExpr_2 + objExpr_3 + objExpr_4
    
    
    """
    Busca_Local(solucao):
    melhora ← verdadeiro
    
    enquanto melhora faça:
        melhora ← falso
        melhor_movimento ← ∅
        melhor_custo ← custo(solucao)
        
        // Explorar vizinhanças
        Para cada cliente c ∈ C:
            Para cada produto p ∈ P:
                Para cada período t ∈ T:
                    
                    // Movimento 1: realocar produção para outro período
                    Para cada período t2 próximo de t:
                        nova_solucao ← mover_producao(solucao, c, p, t, t2)
                        se viável(nova_solucao):
                            custo ← Avaliar(nova_solucao)
                            se custo < melhor_custo:
                                melhor_movimento ← (c,p,t,t2)
                                melhor_custo ← custo
                        
                    // Movimento 2: redistribuir entre clientes
                    Para cada cliente c2 ≠ c:
                        nova_solucao ← transferir_producao(solucao, c → c2, p, t)
                        se viável(nova_solucao):
                            custo ← Avaliar(nova_solucao)
                            se custo < melhor_custo:
                                melhor_movimento ← (c,c2,p,t)
                                melhor_custo ← custo
                                
                    // Movimento 3: reduzir excesso de estoque
                    se estoque[c][p][t] >> demanda[c][p][t]:
                        nova_solucao ← reduzir_estoque(solucao, c,p,t)
                        se viável(nova_solucao):
                            custo ← Avaliar(nova_solucao)
                            se custo < melhor_custo:
                                melhor_movimento ← (reduzir,c,p,t)
                                melhor_custo ← custo
                                
        // Aplicar melhor movimento encontrado
        se melhor_movimento ≠ ∅:
            aplicar(melhor_movimento, solucao)
            melhora ← verdadeiro
    
    retornar solucao

    """
   
