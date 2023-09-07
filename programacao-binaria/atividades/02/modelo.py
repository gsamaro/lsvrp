import gurobipy as gp
from gurobipy import GRB

#================================================================
#FUNÇAO DE LEITURA
#================================================================
def read_qbf_instance(file_path):
    matrix = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        
        N = int(lines[0].split()[0])
        
        for line in lines[1:]:
            row = list(map(int, line.split()))
            matrix.append(row)
    
    return N, matrix

def complete_matrix(triangular_matrix):
    N = len(triangular_matrix)
    complete = [[0] * N for _ in range(N)]
    for i in range(N):
        for j in range(N-i):
            complete[i][j+i] = triangular_matrix[i][j]
            # complete[j+i][i] = triangular_matrix[i][j]
    return complete

#================================================================
#FUNÇAO GUROBI
#================================================================
def gurobi(num_vars,Q):
    # Criar um modelo
    model = gp.Model("MaxQBF")

    # Criar as variáveis binárias
    x = model.addVars(num_vars, vtype=GRB.BINARY, name="x")

    # Definir a função objetivo quadrática
    objective = 0
    for i in range(num_vars):
        for j in range(num_vars):
            objective += Q[i][j] * x[i] * x[j]

    # Adicionar a função objetivo ao modelo (MAXIMIZE pois é um problema de maximização)
    model.setObjective(objective, GRB.MAXIMIZE)

    # for i in range(num_vars-1):
    #     model.addConstr(x[i] + x[i+1] <= 1)

    model.write("modelo.lp")

    # Otimizar o modelo
    model.optimize()

    # Imprimir resultados
    if model.status == GRB.OPTIMAL:
        print("Solução ótima encontrada!")
        print("Valor da função objetivo:", model.objVal)
        for i in range(num_vars):
            print(f"x[{i}] =", int(round(x[i].x)))
    else:
        print("O problema não possui solução ótima.")

#================================================================
#FUNÇAO main
#================================================================
def __main__():
    filename = "./instances/qbf040"
    N, triangular_matrix = read_qbf_instance(filename)
    Q = complete_matrix(triangular_matrix)


    for row in Q:
        print(row)

    gurobi(N,Q)

__main__()