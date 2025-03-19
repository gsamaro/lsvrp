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

    for p in range(len(periods)):
        x=[]
        y=[]
        for v in range(len(periods[p])):
            for i in range(len(periods[p][v])):
                x.append(data['coordXY']['x'][periods[p][v][i]])
                y.append(data['coordXY']['y'][periods[p][v][i]])
            plotar_linha(x,y,titulo=f"Periodo:{p+1}, Veiculo:{v+1}")

    
