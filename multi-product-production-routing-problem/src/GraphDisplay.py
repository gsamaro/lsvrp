import matplotlib.pyplot as plt

def ponto_medio(x1, y1, x2, y2):
    # Calculando as coordenadas do ponto médio
    xm = (x1 + x2) / 2
    ym = (y1 + y2) / 2
    return (xm, ym)

def ploat(x,y,q,r,veicle):
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

    # Gráfico de Barra
    ylabel_barra="Eixo Y (Barra)" 
    titulo_barra="Gráfico de Barra"
    bars = ax2.bar(x, y, color='g', alpha=0.6, width=0.4)
    # ax2.set_xlabel(xlabel)
    ax2.set_ylabel(ylabel_barra, color='g')
    ax2.set_title(titulo_barra)
    ax2.grid(True)

    for bar in bars:
        height = bar.get_height()  # Altura de cada barra
        ax2.text(bar.get_x() + bar.get_width() / 2, height, f'{height}', 
                 ha='center', va='bottom', fontsize=10, color='black')
        
    plt.tight_layout()
    plt.show()

def graphResults(periods = []):
    for period in periods:
        # print(f"Periodo = {period['t']}")
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
            ploat(x,y,q,r,veicle)

