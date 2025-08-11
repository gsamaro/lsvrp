#################################################################################################
# Multi Product Prodction Routing Problem Grasp
# Copyright 2024 Mateus Chacon

# Este programa é um software livre, você pode redistribuí-lo e/ou modificá-lo
# sob os termos da Licença Pública Geral GNU como publicada pela Fundação do Software Livre (FSF),
# na versão 3 da Licença, ou (a seu critério) qualquer versão posterior.

# Este programa é distribuído na esperança de que possa ser útil, mas SEM NENHUMA GARANTIA,
# e sem uma garantia implícita de ADEQUAÇÃO a qualquer MERCADO ou APLICAÇÃO EM PARTICULAR.

# Veja a Licença Pública Geral GNU para mais detalhes
#################################################################################################
from src.log.Logger import Logger
import random
import math
import time
from typing import List, Tuple

import numpy as np

class MultProductProdctionRoutingProblemGrasp:

    def __init__(self,map,dir,log:Logger):
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



    def setMaxInter(self,max_inter):
        self.max_inter = max_inter
    
    def setAlfa(self,alfa):
        self.alfa = alfa
    
    def setSeed(self,seed):
        self.seed = seed


    def getStockInicial(self,t,p,i,estT):
        if(t==0):
            return int(self.I_p_i_0[p][i])
        return int(estT)

    def getDistancesInPeriod(self,candidates_t):
        D = np.zeros((len(candidates_t), len(candidates_t)), dtype=float)
        for i in range(len(candidates_t)):
            for k in range(len(candidates_t)):
                d = self.a_i_k[candidates_t[i]][candidates_t[k]]
                D[i, k] = d
                D[k, i] = d
        return D
    
    def getDemandInPeriod(self,t,candidates_t):
        _i = [[0,0]]
        for i in range(1,len(candidates_t)):
            _p = []
            for p in range(self.p):
                _p.append(self.d_p_i_t[p][candidates_t[i]-1][t])
            _i.append(_p)
        return _i

    def calStockClientInPeriod(self,estT,t):
        candidates_t = [0]
        for i in range(self.i-1):
            for p in range (self.p):
                est_I = self.getStockInicial(t,p,i+1,estT)
                dem = int(self.d_p_i_t[p][i][t])
                calStock = est_I - dem
                #self.log.info(f"{est_I} - {dem} = {calStock}, {i}->{p}")
                if(calStock<0):
                    candidates_t.append(i+1)
                    break
        return candidates_t

    def addDemand(self, load: List[float], demand: List[float]) -> List[float]:
        return [l + d for l, d in zip(load, demand)]

    def canInsertCustomer(self,load: List[float], demand: List[float], capacity: List[float]) -> bool:
        #self.log.info(f"{load} = {sum(load)}")
        for l, d, c in zip(load, demand, capacity):
            #self.log.info(f"{l}+{d} > {c + 1e-9}")
            if sum(load) + d > c + 1e-9:
                return False
        return True


    def routeCost(self,route: List[int], D: np.ndarray) -> float:
        if not route:
            return 0.0
        cost = 0.0
        # assume depot is 0 and included at beginning/end implicitly; route contains customer indices (not depot)
        prev = 0  # depot
        for v in route:
            cost += D[prev, v]
            prev = v
        cost += D[prev, 0]
        return cost

    def twoOptOnRoute(self, route: List[int], D: np.ndarray) -> Tuple[List[int], float]:

        if len(route) <= 2:
            return route, self.routeCost(route, D)
        best = route[:]
        improved = True
        best_cost = self.routeCost(best, D)
        n = len(route)
        while improved:
            improved = False
            for i in range(0, n - 1):
                for j in range(i + 1, n):
                    new_route = best[:i] + best[i:j + 1][::-1] + best[j + 1:]
                    new_cost = self.routeCost(new_route, D)
                    if new_cost + 1e-12 < best_cost:
                        best = new_route
                        best_cost = new_cost
                        improved = True
                        break
                if improved:
                    break
        return best, best_cost

    def greedyRandomizedConstruction(
        self,
        candidates_t,
        demands: List[List[float]],
        capacities: List[List[float]],
        D: np.ndarray,
        n_vehicles: int,
        alpha: float = 0.3,
        rng: random.Random = None,
    ) -> List[List[int]]:
        """
        Constrói uma solução factível (se possível) usando RCL:
        - começa com todas as rotas vazias e cargas zero
        - escolhe clientes não visitados e insere em alguma rota na posição final se couber
        - custo de inserir um cliente c em veículo k = aumento de distância ao colocar c no final
        - RCL com parâmetro alpha
        """
        if rng is None:
            rng = random

        customers = list(range(1, len(demands)))
        unserved = set(customers)

        # inicia cargas e rotas
        loads = [[0.0] * len(capacities[0]) for _ in range(n_vehicles)]
        routes: List[List[int]] = [[] for _ in range(n_vehicles)]

        for v in range(n_vehicles):
            routes[v].append(0)

        # tentativa simples: enquanto houver não-atendidos, escolhe cliente + veículo via RCL
        while unserved:
            candidate_list = []  # tuples (cliente, veiculo, custo_increase)
            for c in list(unserved):
                for k in range(n_vehicles):
                    if self.canInsertCustomer(loads[k], demands[c], capacities[k]):
                        # custo de inserir no final da rota
                        prev = 0 if not routes[k] else routes[k][-1]
                        inc = D[prev, c] + D[c, 0] - D[prev, 0]
                        candidate_list.append((c, k, inc))

            if not candidate_list:
                # não há inserção possível -> tentativa falha (solução inviável)
                # retornamos rotas incompletas para que o GRASP considere a iteração inválida
                return None

            costs = [t[2] for t in candidate_list]
            c_min = min(costs)
            c_max = max(costs)
            threshold = c_min + alpha * (c_max - c_min)
            rcl = [t for t in candidate_list if t[2] <= threshold]
            chosen = rng.choice(rcl)
            cliente, veiculo, _ = chosen

            # insere no final
            routes[veiculo].append(cliente)
            loads[veiculo] = self.addDemand(loads[veiculo], demands[cliente])
            unserved.remove(cliente)

        _v=[]
        for v in range(len(routes)):
            _i = []
            for i in range(len(routes[v])):
                _i.append(candidates_t[routes[v][i]])
            _v.append(_i)
        return _v

    def randomGreedySolution(self):
        
    
        for t in range(self.t):
        
            candidates_t = self.calStockClientInPeriod(0, t)
            
            dem_t = self.getDemandInPeriod(t,candidates_t)
            D = self.getDistancesInPeriod(candidates_t)

            capacities = []
            for _ in range(self.v):
                _p = []
                for p in range(self.p):
                    _p.append(float(self.C))
                capacities.append(_p)


    

            rng = random.Random(self.seed)
            
            routes = self.greedyRandomizedConstruction(candidates_t, dem_t, capacities, D, self.v, self.alfa, rng)

            self.log.info(f"{routes}")

                   







        return 0




    def solver(self,numThreads=None,timeLimit=None):

        
        self.randomGreedySolution()
        return []
    
    def getResults(self):
        return []

   