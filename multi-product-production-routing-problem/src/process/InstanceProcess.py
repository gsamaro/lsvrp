from src.helpers.ReadPrpFile import ReadPrpFile as RD
from src.solvers.MultProductProdctionRoutingProblem import MultProductProdctionRoutingProblem as MPPRP
from src.solvers.MultProductProdctionRoutingProblemGrasp import MultProductProdctionRoutingProblemGrasp as MPPRPG
from src.log.Logger import Logger
from src.process.ProcessResults import getResults
from src.helpers.GraphDisplay import graphResults

class InstanceProcess:

    def __init__(self,instance,output,isPloat='false',numThreads=None,timeLimit=None,log:Logger=None,solver="GRASP"):
        self.instance = instance
        self.isPloat = isPloat
        self.numThreads = numThreads
        self.timeLimit = timeLimit
        self.output=output
        self.isFinished=False
        self.solver = solver
        self.log:Logger = log

    def isProcessFinished(self):
        return self.isFinished
    
    def solverInstancie(self, data):
        match self.solver:
            case "GUROBY":
                self.log.info(f" Solver: GUROBY")
                return MPPRP(data,self.output,self.log)
                
            case "GRASP":
                self.log.info(f" Solver: GRASP")
                return MPPRPG(data,self.output,self.log)
            
            case _: 
                print(" VALOR SETADO COMO DEFAULT ----- SEM SOLVER ")
                return 0

        return 

    def process(self):

        data = RD(file_path=self.instance,log=self.log).getDataSet()
        instance = self.solverInstancie(data)
       
        instance.solver(timeLimit=self.timeLimit,numThreads=self.numThreads)
        Z,X,Y,I,R,Q,FO,GAP,TIME,SOL_COUNT = instance.getResults()

        results = getResults(data,self.output,Z,X,Y,I,R,Q,FO,GAP,TIME,SOL_COUNT)

        if(self.isPloat=='true'):
            graphResults(results['periods'],{'coordsX':data['coordXY']['x'],'coordsY':data['coordXY']['y']},self.output)

        self.isFinished = True
