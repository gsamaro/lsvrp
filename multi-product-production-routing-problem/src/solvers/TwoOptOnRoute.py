#################################################################################################
# Two Opt On Route
# Copyright 2024 Mateus Chacon

# Este programa é um software livre, você pode redistribuí-lo e/ou modificá-lo
# sob os termos da Licença Pública Geral GNU como publicada pela Fundação do Software Livre (FSF),
# na versão 3 da Licença, ou (a seu critério) qualquer versão posterior.

# Este programa é distribuído na esperança de que possa ser útil, mas SEM NENHUMA GARANTIA,
# e sem uma garantia implícita de ADEQUAÇÃO a qualquer MERCADO ou APLICAÇÃO EM PARTICULAR.

# Veja a Licença Pública Geral GNU para mais detalhes
#################################################################################################
from typing import List, Tuple
import numpy as np

class TwoOptOnRoute:

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