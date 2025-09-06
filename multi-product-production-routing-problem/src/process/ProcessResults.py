import json
import os
from src.helpers.Converter import toStopPoint

def getResults(data,dir,Z,X,Y,I,R,Q,FO,GAP,TIME,SOL_COUNT,RELAXED_MODEL_OBJE_VAL, NODE_COUNT, OBJ_BOUND):
    routes = [[toStopPoint(v) for v in Z[t]] for t in range(len(Z))]
    periods = []
    for t in range(len(routes)):
        veicles = []
        for v in range(len(routes[t])):
            points = []
            v_qtd_max = 0
            for i in range(len(routes[t][v])):
                products = []
                r_current = 0

                for p in range (len(Q[t][v])): 
                    if(i!=len(routes[t][v])-1): 
                        r_current=R[t][v][p][routes[t][v][i+1]][routes[t][v][i]]

                    products.append({
                        'p':p+1,
                        'qtd':Q[t][v][p][routes[t][v][i]],
                        'r': r_current
                    })
                    v_qtd_max = v_qtd_max+Q[t][v][p][routes[t][v][i]]
                points.append({
                    'point': routes[t][v][i],
                    'x': data['coordXY']['x'][routes[t][v][i]],
                    'y': data['coordXY']['y'][routes[t][v][i]],
                    'products':products
                })
            veicles.append({
                'v':v+1,
                'points':points,
                'v_qtd_max':v_qtd_max
            })

        productions = []
        for p in range (len(X[t])):
            productions.append({
                'p':p+1,
                'qtd': X[t][p],
                'isProduction': int(Y[t][p])
            })

        est = []
        est_i = []
        dem = []
        for i in range (len(I[t])):
            e_p_current = []
            e_i_p_current = []
            d_p_current = []
            e_current = 0
            e_i_current = 0
            d_current = 0
            for p in range(len(I[t][i])):
                if(t==0):
                    e_i_current = data['I_pi0'][p][i]
                else:
                    e_i_current = I[t-1][i][p]

                e_current = I[t][i][p]

                if(i!=len(I[t])-1):
                    d_current = data['d_pit'][p][i][t]

                e_i_p_current.append({
                    'p':p+1,
                    'qtd':e_i_current
                })
                e_p_current.append({
                    'p':p+1,
                    'qtd':e_current
                })
                d_p_current.append({
                    'p':p+1,
                    'qtd':d_current
                })
            est_i.append({
                'point':i,
                'products': e_i_p_current
            })
            est.append({
                'point':i,
                'products': e_p_current
            })
            dem.append({
                'point':i+1,
                'products': d_p_current
            })

        periods.append({
            't':t+1,
            'veicles':veicles,
            'productions':productions,
            'dem':dem,
            'estq':est,
            'estq_i':est_i
        })

    results = {
        'periods':periods,
        'FO':FO,
        'gap':GAP,
        'time':TIME,
        'solCount':SOL_COUNT,
        'relaxeModelObjeVal':RELAXED_MODEL_OBJE_VAL,
        'nodeCount':NODE_COUNT,
        'objBound':OBJ_BOUND
    }

    caminho_arquivo = os.path.join(dir, "result.json")
    with open(caminho_arquivo, "w", encoding="utf-8") as arquivo:
        json.dump(results, arquivo, indent=4, ensure_ascii=False)
    
    return results

