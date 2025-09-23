from src.helpers.ReadPrpFile import ReadPrpFile as RD
from src.solvers.MultProductProdctionRoutingProblem import MultProductProdctionRoutingProblem as MPPRP
from src.solvers.MultProductProdctionRoutingProblemGreedyConstructiveHeuristic import MultProductProdctionRoutingProblemGreedyConstructiveHeuristic as MPPRPG
from src.log.Logger import Logger
from src.process.ProcessResults import getResults, new_get_results
from src.helpers.GraphDisplay import graphResults
import pdb

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
                return MPPRP(map=data,dir=self.output,log=self.log,start={"start":False})
            
            case "HEURISTICA_CONSTRUTIVA_MIT_START":
                self.log.info(f" Solver: HEURISTICA_CONSTRUTIVA_MIT_START")
                instancia = MPPRPG(map=data,dir=self.output,log=self.log)
                instancia.setMitStart(True)
                return instancia
            
            case "HEURISTICA_CONSTRUTIVA":
                self.log.info(f" Solver: HEURISTICA_CONSTRUTIVA")
                return MPPRPG(map=data,dir=self.output,log=self.log)
            
            case _: 
                print(" VALOR SETADO COMO DEFAULT ----- SEM SOLVER ")
                return 0

        return 

    def process(self):

        data = RD(file_path=self.instance,log=self.log).getDataSet()
        instance = self.solverInstancie(data)
       
        instance.solver(timeLimit=self.timeLimit,numThreads=self.numThreads)

        # Z,X,Y,I,R,Q,FO,GAP,TIME,SOL_COUNT, RELAXED_MODEL_OBJE_VAL, NODE_COUNT, OBJ_BOUND = instance.getResults()
        FO, f1, f2, f3, f4, GAP, TIME, SOL_COUNT, RELAXED_MODEL_OBJE_VAL, NODE_COUNT, OBJ_BOUND = instance.new_get_results()

        # results = getResults(data,self.output,Z,X,Y,I,R,Q,FO,GAP,TIME,SOL_COUNT,RELAXED_MODEL_OBJE_VAL,NODE_COUNT,OBJ_BOUND)

        new_get_results(self.output, FO,f1,f2,f3,f4,GAP,TIME,SOL_COUNT,RELAXED_MODEL_OBJE_VAL,NODE_COUNT,OBJ_BOUND)

        # if(self.isPloat=='true'):
        #     graphResults(results['periods'],{'coordsX':data['coordXY']['x'],'coordsY':data['coordXY']['y']},self.output)

        self.isFinished = True
