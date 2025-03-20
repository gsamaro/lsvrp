import matplotlib.pyplot as plt
import numpy as np

def ponto_medio(x1, y1, x2, y2):
    # Calculando as coordenadas do ponto médio
    xm = (x1 + x2) / 2
    ym = (y1 + y2) / 2
    return (xm, ym)

def ploat(coords,veicle,period):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10))  # 1 linha, 2 colunas
    
    ax1.scatter(coords['coordsX'], coords['coordsY'],  color='red', marker='o', linestyle='-', label="Pontos")

    for i in range(len(veicle['points'])-1):

        x = veicle['points'][i]['x']
        y = veicle['points'][i]['y']
        x_1 = veicle['points'][i+1]['x']
        y_1 = veicle['points'][i+1]['y']
        point = veicle['points'][i]['point']


        ax1.plot([x, x_1], [y, y_1], color='blue', linewidth=2) 

        xm,ym = ponto_medio(x,y,x_1,y_1)
        if(i==0):
            ax1.text(x, y, f"Planta\n({x},{y})", fontsize=10, ha='right', va='bottom')
        else:
            txt = ''
            txt2 = ''
            for p in range(len(veicle['points'][i]['products'])):
                q = veicle['points'][i]['products'][p]['qtd']
                r = veicle['points'][i]['products'][p]['r']
                prd = veicle['points'][i]['products'][p]['p']


                txt=txt + f"p{prd}: {q}\n"
                txt2=txt2 + f"p{prd}: {r}\n"
            ax1.text(x, y, f"{txt}\nPointo:{point}", fontsize=10, ha='right', va='bottom')
            ax1.text(xm, ym, f"{txt2}", fontsize=10, ha='right', va='bottom')

    ax1.set_title(f"Rota Veiculo: {veicle['v']}, Qtd Tranp: {veicle['v_qtd_max']}")
    ax1.grid(True)
    ax1.legend()
# =================================== grafico de bara ==============================================

    x_bar = []
    y_est_bar = []
    y_dem_bar = [] 
    y_est_I_b = []
    for i in range(len(period['estq'])):
        point = period['estq'][i]['point']
        for p in range(len(period['estq'][i]['products'])):
            x_bar.append(f" x_{point}_p{period['estq'][i]['products'][p]['p']}")
            y_est_bar.append(period['estq'][i]['products'][p]['qtd'])
            y_est_I_b.append(period['estq_i'][i]['products'][p]['qtd'])
            if(i==0):
                y_dem_bar.append(0)
            else:
                y_dem_bar.append(period['dem'][i-1]['products'][p]['qtd'])

    # Gráfico de Barra

    titulo_barra=f"Estoque Vs Clinte_Produto: Periodo = {period['t']}"

    largura = 0.4
    x_d = np.arange(len(x_bar)) 
    ax2.bar(x_d - largura/2, y_est_bar, largura, label="Estoque Final", color="blue", align='edge')
    ax2.bar(x_d, y_dem_bar, largura, label="Demanda", color="orange", align='edge')
    ax2.bar(x_d + largura, y_est_I_b, largura, label="Estoque Inicial", color="green")
 
    # ax2.set_xlabel(xlabel)
    ax2.set_title(titulo_barra)
    ax2.grid(True)

    # Adicionar os valores nas barras
    for i in range(len(x_bar)):
        ax2.text(x_d[i] - largura/2, y_est_bar[i] + 0.5, str(y_est_bar[i]), ha='center', color='black', fontsize=12, fontweight='bold')
        ax2.text(x_d[i], y_dem_bar[i] - 10, str(y_dem_bar[i]), ha='center', color='black', fontsize=12, fontweight='bold')
        ax2.text(x_d[i] + largura, y_est_I_b[i] - 20, str(y_est_I_b[i]), ha='center', color='black', fontsize=12, fontweight='bold')

  
    # Ajustar os rótulos no eixo X
    plt.xticks(x_d, x_bar)

    # Melhorar a legenda
    ax2.legend(title="Métricas", loc="upper right", fontsize=12, title_fontsize=14, edgecolor="black")

        
    plt.tight_layout()
    plt.show()

def graphResults(periods = [], coords={}):
    for t in range(len(periods)):
        for veicle in periods[t]['veicles']:
            ploat(coords,veicle,periods[t])


