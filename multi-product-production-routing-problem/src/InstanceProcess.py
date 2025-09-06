from src.helpers.ReadPrpFile import ReadPrpFile as RD
from src.solvers.MultProductProdctionRoutingProblem import MultProductProdctionRoutingProblem as MPPRP
from src.process.ProcessResults import getResults
from src.helpers.GraphDisplay import graphResults

class InstanceProcess:

    def __init__(self,instance,output,isPloat='false',numThreads=None,timeLimit=None):
        self.instance = instance
        self.isPloat = isPloat
        self.numThreads = numThreads
        self.timeLimit = timeLimit
        self.output=output
        self.isFinished=False

    def isProcessFinished(self):
        return self.isFinished

    def process(self):
        data = RD(self.instance).getDataSet()

        mpprp = MPPRP(data,self.output)
        mpprp.solver(timeLimit=self.timeLimit,numThreads=self.numThreads)
        Z,X,Y,I,R,Q,FO,GAP,TIME,SOL_COUNT, RELAXED_MODEL_OBJE_VAL, NODE_COUNT, OBJ_BOUND = mpprp.getResults()

        results = getResults(data,self.output,Z,X,Y,I,R,Q,FO,GAP,TIME,SOL_COUNT,RELAXED_MODEL_OBJE_VAL,NODE_COUNT,OBJ_BOUND)

        if(self.isPloat=='true'):
            graphResults(results['periods'],{'coordsX':data['coordXY']['x'],'coordsY':data['coordXY']['y']},self.output)

        self.isFinished = True
