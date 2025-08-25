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

class GreedyRandomizedConstructionRoute:

    def __init__(self,log:Logger):
        self.log = log

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

        print(n_vehicles)

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

        return [[clients[i] for i in route] for route in routes]
