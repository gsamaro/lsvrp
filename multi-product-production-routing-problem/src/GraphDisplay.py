import matplotlib.pyplot as plt

def ponto_medio(x1, y1, x2, y2):
    # Calculando as coordenadas do ponto médio
    xm = (x1 + x2) / 2
    ym = (y1 + y2) / 2
    return (xm, ym)

def plotar_linha(x, y, titulo="Gráfico de Linha", infs_v = {},xlabel="Eixo X", ylabel="Eixo Y"):
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

    # print(infs_v)
    # Adiciona legendas (labels) para cada ponto

    for i in range(len(x)-1):
        xm,ym = ponto_medio(x[i],y[i],x[i+1],y[i+1])
        if(i==0):
            ax1.text(x[i], y[i], f"Planta\n({x[i]},{y[i]})", fontsize=10, ha='right', va='bottom')
        else:
            txt = ''
            txt2 = ''
            for p in range(len(infs_v['infs_p'])):
                txt=txt + f"p{p+1}: {infs_v['infs_p'][p]['infs_e'][i]}\n"
                txt2=txt2 + f"p{p+1}: {infs_v['infs_p'][p]['infs_i'][i]}\n"
            ax1.text(x[i], y[i], f"{txt}", fontsize=10, ha='right', va='bottom')
            ax1.text(xm, ym, f"{txt2}", fontsize=10, ha='right', va='bottom')

    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    ax1.set_title(f"{titulo}, Quantidade Total Produtos: {infs_v['v_total']}")
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
        
    plt.tight_layout()
    plt.show()