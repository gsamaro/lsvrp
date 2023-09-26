import math
from gurobipy import Model, GRB
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def heuristica_gulosa(circle_radii):
    def distancia(p1, p2):
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def cabe_no_container(circulo, pos, outros_circulos):
        for outro in outros_circulos:
            if distancia(pos, outro) < circulo + outros_circulos[outro]:
                return False
        return True

    def posicionar_circulo(circulo, outros_circulos, R):
        tentativas = [(0, R), (R, 0), (0, -R), (-R, 0)]
        for t in tentativas:
            if cabe_no_container(circulo, t, outros_circulos):
                return t
        return None

    circle_radii = sorted(circle_radii, reverse=True)
    circles_posicionados = {(0,0): circle_radii[0]}
    R = circle_radii[0]

    for r in circle_radii[1:]:
        pos = posicionar_circulo(r, circles_posicionados, R)
        while not pos:
            R += 0.1
            pos = posicionar_circulo(r, circles_posicionados, R)
        circles_posicionados[pos] = r

    return R, circles_posicionados

# Parâmetros
circle_radii = [1, 2, 3, 4, 5, 6, 7]  

circles = [(i, r) for i, r in enumerate(circle_radii)]

print(circles)

# Usar heurística gulosa para obter ponto inicial
R_inicial, posicoes_iniciais = heuristica_gulosa(circle_radii)

print(R_inicial)

# Criar modelo
m = Model("miqcp_example")

# Variáveis de decisão
x = {}
y = {}
for c, r in circles:
    x[c] = m.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"x_{c}")
    y[c] = m.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"y_{c}")
R = m.addVar(lb=R_inicial, ub=GRB.INFINITY, name="R")

# Configurar ponto inicial
for c, r in circles:
    pos = list(posicoes_iniciais.keys())[c] # pegue a posição da heurística
    x[c].start = pos[0]
    y[c].start = pos[1]

# Restrições
for c, r in circles:
    m.addConstr((x[c] * x[c]) + (y[c] * y[c]) <= (R - r) * (R - r))
for (i, ri) in circles:
    for (j, rj) in circles:
        if i < j:
            m.addConstr((x[i] - x[j]) * (x[i] - x[j]) + (y[i] - y[j]) * (y[i] - y[j]) >= (ri + rj) * (ri + rj))

# Objetivo
m.setObjective(R, GRB.MINIMIZE)

# Parâmetros Gurobi
m.setParam('TimeLimit', 1800)
m.setParam('NonConvex', 2)

# Função para plotar os círculos
def plot_circles(circles, container_radius):
    fig, ax = plt.subplots()
    
    # Plotando o contêiner
    container_circle = patches.Circle((0, 0), container_radius, fc='none', ec='blue', linestyle='--')
    ax.add_patch(container_circle)
    
    for (x, y, r) in circles:
        circle = patches.Circle((x, y), r, fc='none', ec='black')
        ax.add_patch(circle)
        ax.plot(x, y, 'ro')  # plota o centro do círculo

    ax.set_aspect('equal', adjustable='box')
    ax.relim()
    ax.autoscale_view()
    plt.show()

# Resolver modelo
m.optimize()

# Resultados
if m.status == GRB.OPTIMAL:
    print(f"Raio mínimo do contêiner: {R.x}")
    circles_to_plot = []
    for c, r in circles:
        print(f"Círculo com raio {r}: Centro em ({x[c].x}, {y[c].x})")
        circles_to_plot.append((x[c].x, y[c].x, r))
    # Plota os círculos
    plot_circles(circles_to_plot, R.x)

elif m.status == GRB.TIME_LIMIT:
    print("Limite de tempo alcançado!")
    if m.SolCount > 0:
        print(f"Melhor solução encontrada até agora: Raio = {R.x}")
        circles_to_plot = []
        for c, r in circles:
            print(f"Círculo com raio {r}: Centro em ({x[c].x}, {y[c].x})")
            circles_to_plot.append((x[c].x, y[c].x, r))
        # Plota os círculos
        plot_circles(circles_to_plot, R.x)
else:
    print("Nenhuma solução encontrada")