from src.process.WorkerProcess import WorkerProcess
from src.log.Logger import Logger
from src.utils import processInstance   

import json
import shutil
import os

try:
    from mpi4py import MPI
    from mpi4py.futures import MPIPoolExecutor

    MPI_BOOL = True
except:
    print("mpi4py not running")
    MPI_BOOL = False

if __name__ == "__main__":

    with open("config.json", "r") as f:
        config = json.load(f)

    threadsLimitSolver = 0
    if config["solver"]["threadsLimit"] == "None":
        threadsLimitSolver = None
    else:
        threadsLimitSolver = config["solver"]["threadsLimit"]

    timeLimitSolver = config["solver"]["timeLimit"]
    timeSupervisor = config["workers"]["timeSupervisor"]
    workers = config["workers"]["num"]
    output = config["instance"]["output"]
    isPloat = config["instance"]["is_plot"]
    dir = config["instance"]["dir"]
    files = config["instance"]["files"]
    method = config["solver"]["method"]

    if os.path.exists(f"{output}/logs"):
        shutil.rmtree(f"{output}/logs")

    log = Logger(log_dir=f"{output}logs", log_file=f"Worker_0.log", worker_id=0, task=0)

    datas = []
    for file in files:
        if ".dat" in file:
            partes = file.split("/")
            datas.append({"data": partes[0], "files": [partes[1]]})
        else:
            datas.append(
                {
                    "data": file,
                    "files": [
                        f
                        for f in os.listdir(dir + file)
                        if os.path.isfile(os.path.join(dir + file, f))
                    ],
                }
            )

    instancies = []
    for data in datas:
        for file in data["files"]:

            outFile = f"{output}{data['data']}/{file[:-4]}/"
            # if os.path.exists(outFile):
            #     shutil.rmtree(outFile)
            os.makedirs(outFile, exist_ok=True)

            instancies.append(
                {
                    "file": f"{dir}{data['data']}/{file}",
                    "output": outFile,
                    "isPloat": isPloat,
                    "numThreads": threadsLimitSolver,
                    "timeLimit": timeLimitSolver,
                }
            )

    if not MPI_BOOL:
        WorkerProcess(
            workers, timeSupervisor, {"instancia": log, "dirLogs": f"{output}logs"}
        ).process(instancies=instancies, solver=method)
    else:
        with MPIPoolExecutor() as executor:
            try:
                futures = executor.startmap(
                    processInstance,
                    [
                        (instancie, log, w1, w2, w3, w4)
                        for instancie in instancies
                        for w1, w2, w3, w4 in zip(range(1, 2), range(1, 2), range(1, 2), range(1, 2))],
                )
                final_results.append(futures)
                executor.shutdown(wait=True)
            except Exception as e:
                log.error(
                    f"Ocorreu um erro inesperado: {e}: stack: {traceback.format_exc()}"
                )

    """
Explored 11164 nodes (448772 simplex iterations) in 30.82 seconds (21.64 work units)
Thread count was 1 (of 8 available processors)

Solution count 10: 200744 200918 201025 ... 205818

Explored 8394 nodes (406720 simplex iterations) in 27.97 seconds (19.82 work units)
Thread count was 1 (of 8 available processors)

Solution count 10: 200744 201085 201197 ... 203144


"""
