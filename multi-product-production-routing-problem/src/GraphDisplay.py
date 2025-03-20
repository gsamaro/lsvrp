import matplotlib.pyplot as plt
import numpy as np

def ponto_medio(x1, y1, x2, y2):
    # Calculando as coordenadas do ponto médio
    xm = (x1 + x2) / 2
    ym = (y1 + y2) / 2
    return (xm, ym)

def ploat(x,y,q,r,veicle,period):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10))  # 1 linha, 2 colunas
    
    ax1.plot(x, y, marker='o', linestyle='-', color='b', label="Linha")

    for i in range(len(q)-1):
        xm,ym = ponto_medio(x[i],y[i],x[i+1],y[i+1])
        if(i==0):
            ax1.text(x[i], y[i], f"Planta\n({x[i]},{y[i]})", fontsize=10, ha='right', va='bottom')
        else:
            txt = ''
            txt2 = ''
            for p in range(len(q[i])):
                txt=txt + f"p{p+1}: {q[i][p]}\n"
                txt2=txt2 + f"p{p+1}: {r[i][p]}\n"
            ax1.text(x[i], y[i], f"{txt}", fontsize=10, ha='right', va='bottom')
            ax1.text(xm, ym, f"{txt2}", fontsize=10, ha='right', va='bottom')

    ax1.set_title(f"Rota Veiculo: {veicle['v']}, Qtd Tranp: {veicle['v_qtd_max']}")
    ax1.grid(True)
    ax1.legend()
# =================================== grafico de bara ==============================================

    x_bar = []
    y_est_bar = []
    y_dem_bar = [] 
    for i in range(len(period['estq'])):
        point = period['estq'][i]['point']
        for p in range(len(period['estq'][i]['products'])):
            x_bar.append(f" x_{point}_p{period['estq'][i]['products'][p]['p']}")
            y_est_bar.append(period['estq'][i]['products'][p]['qtd'])
            if(i==0):
                y_dem_bar.append(0)
            else:
                y_dem_bar.append(period['dem'][i-1]['products'][p]['qtd'])

    # Gráfico de Barra

    titulo_barra=f"Estoque Vs Clinte_Produto: Periodo = {period['t']}"

    largura = 0.4
    x_d = np.arange(len(x_bar)) 
    ax2.bar(x_d - largura/2, y_est_bar, largura, label="Estoque", color="blue", align='edge')
    ax2.bar(x_d, y_dem_bar, largura, label="Demanda", color="orange", align='edge')

    # ax2.set_xlabel(xlabel)
    ax2.set_title(titulo_barra)
    ax2.grid(True)

    # Adicionar os valores nas barras
    for i in range(len(x_bar)):
        ax2.text(x_d[i] - largura/2, y_est_bar[i] + 0.5, str(y_est_bar[i]), ha='center', color='black', fontsize=12, fontweight='bold')
        ax2.text(x_d[i], y_dem_bar[i] - 10, str(y_dem_bar[i]), ha='center', color='black', fontsize=12, fontweight='bold')
  
    # Ajustar os rótulos no eixo X
    plt.xticks(x_d, x_bar)

    # Melhorar a legenda
    ax2.legend(title="Métricas", loc="upper right", fontsize=12, title_fontsize=14, edgecolor="black")

        
    plt.tight_layout()
    plt.show()

def graphResults(periods = []):
    for period in periods:
        # print(f"Periodo = {period['t']}")
        # print(f"Periodo = {period['productions']}")
        # print(f"Periodo = {period['estq']}")
        for veicle in period['veicles']:
            # print(f"Veiculo = {veicle['v']}")
            x=[]
            y=[]
            q=[]
            r=[]
            for point in veicle['points']:
                # print(f"x={point['x']} -> y={point['y']}")
                x.append(point['x'])
                y.append(point['y'])
                q_p = []
                r_p = []
                for product in point['products']:
                    # print(f"{product['p']} => qtd{product['qtd']} => r {product['r']}")
                    q_p.append(product['qtd'])
                    r_p.append(product['r'])
                q.append(q_p)
                r.append(r_p)
            ploat(x,y,q,r,veicle,period)

