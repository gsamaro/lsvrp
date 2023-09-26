import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from gurobipy import Model, GRB

circle_radii = [1, 2, 3, 4, 5, 6, 7]  

def spiral_heuristic(radii):
    positions = [(0, 0)]  # Comece colocando o maior círculo no centro.
    R = max(radii)  # raio inicial do círculo maior
    
    theta = 0.5  # Ângulo inicial para a espiral.
    for idx in range(1, len(radii)):
        placed = False
        r_offset = 0.1
        while not placed:
            r = R + radii[idx] + r_offset  # Distância do centro do círculo maior.
            x = r * np.cos(theta)
            y = r * np.sin(theta)

            # Verifica se o círculo se sobrepõe a qualquer círculo já colocado.
            overlap = any(np.sqrt((x - px)**2 + (y - py)**2) < radii[idx] + radii[positions.index((px, py))] for px, py in positions)
            
            if not overlap:
                positions.append((x, y))
                placed = True
            r_offset += 0.1
            theta += 0.1  # Incrementa o ângulo para continuar a espiral.

    R_final = max(np.sqrt(x**2 + y**2) + radii[i] for i, (x, y) in enumerate(positions))
    return {pos: radii[i] for i, pos in enumerate(positions)}, R_final

# Usar heurística em espiral para obter ponto inicial
positions_inicial, R_inicial = spiral_heuristic(circle_radii)

def plot_circles(circles, container_radius):
    fig, ax = plt.subplots()
    
    # Plotando o contêiner
    container_circle = patches.Circle((0, 0), container_radius, fc='none', ec='blue', linestyle='--')
    ax.add_patch(container_circle)
    
    for (x, y), r in circles.items():
        circle = patches.Circle((x, y), r, fc='none', ec='black')
        ax.add_patch(circle)
        ax.plot(x, y, 'ro')  # plota o centro do círculo

    ax.set_aspect('equal', adjustable='box')
    ax.relim()
    ax.autoscale_view()
    plt.show()

# Criar modelo
m = Model("miqcp_example")

# Variáveis de decisão
x = {}
y = {}
for idx, r in enumerate(circle_radii):
    x[idx] = m.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"x_{idx}")
    y[idx] = m.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"y_{idx}")

R = m.addVar(lb=0, ub=GRB.INFINITY, name="R", obj=1.0)  # Configure o objetivo aqui

# Configurar ponto inicial
for idx, r in enumerate(circle_radii):
    pos = list(positions_inicial.keys())[idx]  # pegue a posição da heurística
    x[idx].start = pos[0]
    y[idx].start = pos[1]
R.start = R_inicial

# Restrições
for i in range(len(circle_radii)):
    m.addConstr((x[i] * x[i]) + (y[i] * y[i]) <= (R - circle_radii[i]) * (R - circle_radii[i]))

for i in range(len(circle_radii)):
    for j in range(i+1, len(circle_radii)):
        m.addConstr((x[i] - x[j]) * (x[i] - x[j]) + (y[i] - y[j]) * (y[i] - y[j]) >= (circle_radii[i] + circle_radii[j]) * (circle_radii[i] + circle_radii[j]))

# Parâmetros Gurobi
m.setParam('TimeLimit', 1800)
m.setParam('NonConvex', 2)

# Resolver modelo
m.optimize()
