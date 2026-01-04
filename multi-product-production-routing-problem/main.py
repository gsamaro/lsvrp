import json
import os
import shutil

import pandas as pd
from config import Config
from src.helpers.Outputs import _union_results
from src.log.Logger import Logger
from src.process.PostProcessingProcess import PostProcessingProcess
from src.process.WorkerProcess import WorkerProcess

if __name__ == "__main__":

    config = Config.get_nested("solver", "threadsLimit")
    if config == "None":
        threadsLimitSolver = None
    else:
        threadsLimitSolver = config

    config = Config.get_nested("solver", "timeLimit")
    timeLimitSolver = int(config)

    config = Config.get_nested("workers", "timeSupervisor")
    timeSupervisor = int(config)

    config = Config.get_nested("workers", "num")
    workers = config

    config = Config.get_nested("instance", "output")
    output = config

    config = Config.get_nested("instance", "is_plot")
    isPloat = config

    config = Config.get_nested("instance", "dir")
    dir = config

    config = Config.get_nested("instance", "files")
    files = config

    config = Config.get_nested("solver", "method")
    method = config

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

    WorkerProcess(
        workers, timeSupervisor, {"instancia": log, "dirLogs": f"{output}logs"}
    ).run_parallel(instancies=instancies, solver=method)

    postprocessing = PostProcessingProcess(log=log, output=output)
    postprocessing.union_results()
    if Config.get_nested("postprocessing", "build_target"):
        postprocessing.build_target()

    """
Explored 11164 nodes (448772 simplex iterations) in 30.82 seconds (21.64 work units)
Thread count was 1 (of 8 available processors)

Solution count 10: 200744 200918 201025 ... 205818

Explored 8394 nodes (406720 simplex iterations) in 27.97 seconds (19.82 work units)
Thread count was 1 (of 8 available processors)

Solution count 10: 200744 201085 201197 ... 203144


"""
