#################################################################################################
# Greedy Randomized Construction Route
# Copyright 2024 Mateus Chacon

# Este programa é um software livre, você pode redistribuí-lo e/ou modificá-lo
# sob os termos da Licença Pública Geral GNU como publicada pela Fundação do Software Livre (FSF),
# na versão 3 da Licença, ou (a seu critério) qualquer versão posterior.

# Este programa é distribuído na esperança de que possa ser útil, mas SEM NENHUMA GARANTIA,
# e sem uma garantia implícita de ADEQUAÇÃO a qualquer MERCADO ou APLICAÇÃO EM PARTICULAR.

# Veja a Licença Pública Geral GNU para mais detalhes
#################################################################################################
from typing import List
import numpy as np
import random
import pdb
from src.log.Logger import Logger
from src.solvers.TwoOptOnRoute import TwoOptOnRoute

class GreedyRandomizedConstructionRoute:

    def __init__(self,log:Logger):
        self.log = log
        self.twoOpt = TwoOptOnRoute()

    def addDemand(self, load: List[float], demand: List[float]) -> List[float]:
        return [l + d for l, d in zip(load, demand)]
    
    def canInsertCustomer(self, load: List[float], demand: List[float], capacity: List[float]) -> bool:
        """
        Verifica se é possível inserir um cliente respeitando as capacidades.
        """
        total_load = sum(load)

        for d, c in zip(demand, capacity):
            if total_load + d > c + 1e-9:
                return False
        return True
    
    def total_demanda(self, solution: List[List[int]], demands: List[List[float]], c:List[int]) -> float:
        p_size = len(demands[0])
        vehicles = []

        for v in range(len(solution)):
            d = []
            d_t = [0.0 for _ in range(p_size)]
            # Somar demandas atendidas por veículo
            for i in range(len(solution[v])):
                client = solution[v][i]
                d_p_i = []
                for p in range(len(demands[client])):
                    d_t[p] += demands[client][p]
                    d_p_i.append(demands[client][p])
                d.append(d_p_i)
            
            # Atualizar demandas restantes
            d_e = []
            for d_i in range(len(d)):
                d_e_p = []
                for p in range(len(d[d_i])):
                    d_t[p] -= d[d_i][p]
                    d_e_p.append({'produto':p,'restante_veiculo':d_t[p], 'qte_entregue': d[d_i][p]})
                d_e.append({'cliente': c[v][d_i], 'produtos': d_e_p})

            vehicles.append({'veiculo':v, 'entregas': d_e})

        return vehicles
    
    def total_cost(self, solution: List[List[int]], D: np.ndarray) -> float:
        return sum(self.route_cost(r, D) for r in solution)
    
    def route_cost(self, route: List[int], D: np.ndarray) -> float:
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

    def greedyRandomizedConstruction(
        self,
        clients,
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
                self.log.warning("não há inserção possível -> tentativa falha (solução inviável)")
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


        cost=0
        c=[]
        for route in routes:
            route_op, cost_route = self.twoOpt.twoOptOnRoute(route,D)
            cost+=cost_route

            c.append([clients[i] for i in route_op])


        #c = [[clients[i] for i in route] for route in routes]

        return c, self.total_cost(routes,D), self.total_demanda(routes,demands,c)
