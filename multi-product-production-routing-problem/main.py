import sys
import os
from src.ReadPrpFile import ReadPrpFile as RD
from src.MultProductProdctionRoutingProblem import MultProductProdctionRoutingProblem as MPPRP
from src.ProcessResults import getResults
from src.GraphDisplay import graphResults

if __name__ == "__main__":

    past = sys.argv[1]
    file = sys.argv[2]
    isPlot = sys.argv[3]
    dir = f'./out/{past}/{file}/'
    os.makedirs(dir, exist_ok=True)

    data = RD("./data/"+past+"/"+file+".dat").getDataSet()

    mpprp = MPPRP(data,dir)
    mpprp.solver()
    Z,X,Y,I,R,Q,FO,GAP = mpprp.getResults()

    results = getResults(data,dir,Z,X,Y,I,R,Q,FO,GAP)

    if(isPlot=='true'):
        graphResults(results['periods'],{'coordsX':data['coordXY']['x'],'coordsY':data['coordXY']['y']},dir)