def toStopPoint(matriz):
    n = len(matriz)  # número de pontos (dimensão da matriz)
    visitado = [False] * n  # Para marcar quais pontos já visitamos
    lista_paradas = []  # Lista onde vamos construir o caminho
    atual = 0  # Começamos no ponto 0 (ou qualquer outro ponto arbitrário)

    while len(lista_paradas) < n:
        lista_paradas.append(atual)
        visitado[atual] = True
        for prox in range(n):
            if matriz[atual][prox] == 1 and not visitado[prox]:
                atual = prox
                break
    return lista_paradas

def inveterMatrix(matriz):
    n = len(matriz)
    matriz_invertida = [[0] * n for _ in range(n)]

    for i in range(n):
        for j in range(n):
            matriz_invertida[i][j] = abs(matriz[i][j])
    return matriz_invertida