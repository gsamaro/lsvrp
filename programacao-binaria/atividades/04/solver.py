import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt
import numpy as np
import json

def circle_packing_gurobi(radii = []):
    num_circles = len(radii)

    # Crie o modelo
    model = gp.Model("PECD")

    # Defina as variáveis de decisão
    x = model.addVars(num_circles, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="x")
    y = model.addVars(num_circles, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="y")
    # Defina o raio do circulo maior (variável de decisão)
    R = model.addVar(lb=0, name="R")

    # Adicione restrições para evitar sobreposição
    for i in range(num_circles):
        for j in range(num_circles):
            if(i!=j):
                model.addConstr((x[i] - x[j])**2 + (y[i] - y[j])**2 >= (radii[i] + radii[j])**2)

    # Adicione restrições para manter os círculos dentro do recipiente
    for i in range(num_circles):
        model.addConstr((x[i])**2 + (y[i])**2  <= ( R - radii[i])**2) 

    #O raio não pode ser 0
    model.addConstr(R>=1)

    # Defina a função objetivo para minimizar o raio do recipiente
    model.setObjective(R, sense=GRB.MINIMIZE)

    # Otimizar o modelo
    model.Params.NonConvex = 2      # Aceitar termos não convexos
    model.Params.TimeLimit = 10800   # 1 hora de execução
    # gera o lp do modelo
    lp = "model-LP.mps"
    model.write(lp)
    model.optimize()

    points = []
    for i in range(num_circles):
        point = {
            "x":x[i].x,
            "y":y[i].x,
            "r":radii[i]
        }
        points.append(point)

    return {
        "points": points,
        "R": R.x,
        "R_x": 0,
        "R_y": 0,
        "fo": model.ObjVal,
        "gap": model.MIPGap
    }


def plot_solution(awnser={},name =""):
    lim = awnser["R"]
    centros = [(awnser["R_x"],awnser["R_y"])]
    raios = [awnser["R"]]

    for i in range( len(awnser['points'])):
        centros.append( (awnser['points'][i]['x'],awnser['points'][i]['y']) )
        raios.append(awnser['points'][i]['r'])

    # Gerar o gráfico
    fig, ax = plt.subplots()

    # Plotar cada círculo
    for i in range(len(centros)):
        circle = plt.Circle(centros[i], raios[i], fill=False)
        ax.add_patch(circle)

    # Configurações do gráfico
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect('equal', adjustable='box')  # Para garantir que os círculos mantenham a forma correta

    # Exibir o gráfico
    plt.title('Plotagem de Círculos')
    plt.savefig("./"+name+'.png')
   
    lp = "./"+name+".json"
    with open(lp, 'a') as arquivo:
        arquivo.write(json.dumps(awnser)) 

# 1 - 7  - 1,2,. . . ,7
# 2 - 15 - 1, 2 e 3 (5× cada)
# 3 - 30 - 1, 2 e 3 (10× cada)
# 4 - 50 - 1, 2, 3, 4 e 5 (10× cada)
# 5 - 100 - 1,2,. . . ,10 (10× cada)
def __main__():
    # radii = [1,1]
    # awnser = circle_packing_gurobi(radii=radii)
    # print(awnser)
    # plot_solution(awnser,"test")

    # radii = [1,2,3,4,5,6,7]
    # awnser = circle_packing_gurobi(radii=radii)
    # print(awnser)
    # plot_solution(awnser,"instancia_1")

    # radii = [1]*5 + [2]*5 + [3]*5
    # awnser = circle_packing_gurobi(radii=radii)
    # print(awnser)
    # plot_solution(awnser,"instancia_2")

    radii = [1]*10 + [2]*10 + [3]*10
    awnser = circle_packing_gurobi(radii=radii)
    print(awnser)
    plot_solution(awnser,"instancia_3")

    radii = [1]*10 + [2]*10 + [3]*10 + [4]*10 + [5]*10
    awnser = circle_packing_gurobi(radii=radii)
    print(awnser)
    plot_solution(awnser,"instancia_4")

    radii = [1]*10 + [2]*10 + [3]*10 + [4]*10 + [5]*10 + [6]*10 + [7]*10 + [8]*10 + [9]*10 + [10]*10
    awnser = circle_packing_gurobi(radii=radii)
    print(awnser)
    plot_solution(awnser,"instancia_5")
__main__()