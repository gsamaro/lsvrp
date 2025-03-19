import sys
from src.ReadPrpFile import ReadPrpFile as RD
from src.MultProductProdctionRoutingProblem import MultProductProdctionRoutingProblem as MPPRP
from src.Converter import toStopPoint
from src.GraphDisplay import plotar_linha

if __name__ == "__main__":

    data = sys.argv[1]
    file = sys.argv[2]

    data = RD("./data/"+data+"/"+file+".dat").getDataSet()

    mpprp = MPPRP(data)
    mpprp.solver()
    Z,X,Y,I,R,Q,FO,GAP = mpprp.getResults()

    periods = [[toStopPoint(v) for v in Z[t]] for t in range(len(Z))]

    print(periods)
    for t in range(len(periods)):
        x=[]
        y=[]

        print("Periodo :",t+1)
        for v in range(len(periods[t])):
            infs_p = []
            v_total = 0
            for p in range (len(R[t][v])):
                # print(f"============== p{p} ==========================")
                # print(R[t][v][p])

               

                infs_e = []
                p_current = 0
                p_max = 0
                for i in range(len(periods[t][v])):
                    p_current = Q[t][v][p][periods[t][v][i]]
                    p_max=p_max+p_current
                    infs_e.append(p_current)
                v_total = v_total + p_max



                infs_i = []
                p_current = 0
                for i in range(0,len(periods[t][v])-1):
                    # print(f"{periods[t][v][i+1]} -> {periods[t][v][i]} = { R[t][v][p][periods[t][v][i+1]][periods[t][v][i]] }") 
                    p_current = R[t][v][p][periods[t][v][i+1]][periods[t][v][i]]
                    infs_i.append(p_current)

                infs_p.append({'infs_i':infs_i,'p_max':p_max, 'infs_e': infs_e})
                
            infs_v = {
                'infs_p':infs_p,
                'v_total':v_total
            }
            for i in range(len(periods[t][v])):
                x.append(data['coordXY']['x'][periods[t][v][i]])
                y.append(data['coordXY']['y'][periods[t][v][i]])
              
            plotar_linha(x,y,titulo=f"Periodo:{p+1}, Veiculo:{v+1}",infs_v = infs_v)

    
