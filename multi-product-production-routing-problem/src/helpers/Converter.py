import numpy as np

def matriz_to_route(matriz):
    n = matriz.shape[0]
    rota = []
    visitado = set()
    atual = 0  # Começa no nó 0
    
    while atual not in visitado:
        rota.append(int(atual))
        visitado.add(atual)
        prox = np.where(matriz[atual] == 1)[0]  # Agora verifica corretamente na matriz original
        if len(prox) == 0:
            break  # Se não houver próximo, encerra
        atual = prox[0]  # Pega o primeiro próximo disponível
    
    rota.append(0)  # Volta ao início
    return rota

def inverter_matriz(matriz):
    return  np.array(matriz).T  # Transposta da matriz

def toStopPoint(matriz):
    return matriz_to_route(inverter_matriz(matriz))
