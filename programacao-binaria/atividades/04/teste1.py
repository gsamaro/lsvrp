from gurobipy import Model, GRB, quicksum
import matplotlib.pyplot as plt
import numpy as np

def circle_packing_gurobi():
    m = Model()
    
    num_circles = 5
    radii = [1]*5 + [2]*5 + [3]*5

    print(radii)
    
    # Variáveis
    x = m.addVars(num_circles, vtype=GRB.CONTINUOUS, name="x")
    y = m.addVars(num_circles, vtype=GRB.CONTINUOUS, name="y")
    R = m.addVar(vtype=GRB.CONTINUOUS, name="R")
    
    # Restrições
    for i in range(num_circles):
        m.addQConstr(x[i]*x[i] + y[i]*y[i] <= (R - radii[i]) * (R - radii[i]))
        
        for j in range(i+1, num_circles):
            m.addQConstr((x[i] - x[j]) * (x[i] - x[j]) + (y[i] - y[j]) * (y[i] - y[j]) >= (radii[i] + radii[j]) * (radii[i] + radii[j]))
    
    m.setObjective(R, GRB.MINIMIZE)
    
    m.Params.TimeLimit = 45*60  # 45 minutos
    m.Params.NonConvex = 2      # Aceitar termos não convexos
    m.optimize()
    
    return m, x, y, R
def plot_solution(x, y, radii, R):
    plt.figure(figsize=(10,10))
    plt.gca().add_patch(plt.Circle((0, 0), R.X, fc='y'))
    
    for i in range(len(radii)):
        plt.gca().add_patch(plt.Circle((x[i].X, y[i].X), radii[i], fc='r'))
        
    plt.xlim(-R.X-1, R.X+1)
    plt.ylim(-R.X-1, R.X+1)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

m, x, y, R = circle_packing_gurobi()
radii = [1]*5 + [2]*5 + [3]*5
plot_solution(x, y, radii, R)
print(f"Menor raio possível do círculo maior: {R.X}")