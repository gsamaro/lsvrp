from src.WorkerProcess import WorkerProcess
from src.TablesResult import TablesResult
import json
import shutil
import os

if __name__ == "__main__":

    with open('config.json', 'r') as f:
        config = json.load(f)

    threadsLimitSolver=0
    if(config['solver']['threadsLimit']=="None"):
        threadsLimitSolver = None
    else:
        threadsLimitSolver=config['solver']['threadsLimit']

    timeLimitSolver = config['solver']['timeLimit']
    timeSupervisor = config['workers']['timeSupervisor']
    workers = config['workers']['num']
    output = config['instance']['output']
    isPloat = config['instance']['is_plot']
    dir = config['instance']['dir']
    files = config['instance']['files']

    logs = f"{output}/logs"
    if os.path.exists(logs):
        shutil.rmtree(logs)

    datas = []
    for file in files:
        if ".dat" in file:
            partes = file.split("/")
            datas.append({
                'data': partes[0],
                'files': [partes[1]]
            })
        else:
            datas.append({
                'data': file,
                'files': [f for f in os.listdir(dir+file) if os.path.isfile(os.path.join(dir+file, f))]
            })

    instancies = []
    for data in datas:
        for file in data['files']:

            outFile = f"{output}{data['data']}/{file[:-4]}/"
            # if os.path.exists(outFile):
            #     shutil.rmtree(outFile)
            # os.makedirs(outFile, exist_ok=True) 

            instancies.append({
                'file': f"{dir}{data['data']}/{file}",
                'output': outFile,
                'isPloat':isPloat,
                'numThreads':threadsLimitSolver,
                'timeLimit':timeLimitSolver
            })





    WorkerProcess(workers,timeSupervisor,dirLogs=f'{output}logs').process(instancies = instancies)

    # TablesResult.process(datas=datas)






    