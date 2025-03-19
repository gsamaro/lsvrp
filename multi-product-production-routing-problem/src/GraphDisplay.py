import matplotlib.pyplot as plt

def plotar_linha(x, y, titulo="Gráfico de Linha", xlabel="Eixo X", ylabel="Eixo Y"):
    """
    Plota um gráfico de linha com os pontos (x, y).

    Parâmetros:
    - x: Lista ou array com valores do eixo X.
    - y: Lista ou array com valores do eixo Y.
    - titulo: Título do gráfico.
    - xlabel: Nome do eixo X.
    - ylabel: Nome do eixo Y.

    Retorno:
    - Exibe o gráfico na tela.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10))  # 1 linha, 2 colunas

    ax1.plot(x, y, marker='o', linestyle='-', color='b', label="Linha")
    # Adiciona legendas (labels) para cada ponto
    for i in range(len(x)):
        ax1.text(x[i], y[i], f"({x[i]}, {y[i]})", fontsize=10, ha='right', va='bottom')
    
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    ax1.set_title(titulo)
    ax1.grid(True)
    ax1.legend()

    # Gráfico de Barra
    ylabel_barra="Eixo Y (Barra)" 
    titulo_barra="Gráfico de Barra"
    bars = ax2.bar(x, y, color='g', alpha=0.6, width=0.4)
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel(ylabel_barra, color='g')
    ax2.set_title(titulo_barra)
    ax2.grid(True)

    for bar in bars:
        height = bar.get_height()  # Altura de cada barra
        ax2.text(bar.get_x() + bar.get_width() / 2, height, f'{height}', 
                 ha='center', va='bottom', fontsize=10, color='black')

    # Ajusta o layout para evitar sobreposição
    plt.tight_layout()
    plt.show()