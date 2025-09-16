from src.process.InstanceProcess import InstanceProcess

def processInstance(instancie, log, w1, w2, w3, w4):
    InstanceProcess(
        instancie["instancie"]["file"],
        instancie["instancie"]["output"],
        isPloat=instancie["instancie"]["isPloat"],
        timeLimit=instancie["instancie"]["timeLimit"],
        numThreads=instancie["instancie"]["numThreads"],
        log=log,
        solver="GUROBY",
    ).process()
