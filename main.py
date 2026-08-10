import json
import os
import shutil
import subprocess
from datetime import datetime
from hashlib import sha1

import pandas as pd
from config import Config
from src.helpers.Outputs import _union_results
from src.helpers.JobGuardrails import JobGuardrails
from src.log.Logger import Logger
from src.process.PostProcessingProcess import PostProcessingProcess
from src.process.WorkerProcess import WorkerProcess


def _get_git_commit_hash6():
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()[:6]
    except Exception:
        return "000000"


def _build_run_tag(now: datetime):
    date_part = now.strftime("%Y-%m-%d")
    commit_part = _get_git_commit_hash6()
    ts_full = now.astimezone().isoformat()
    ts_part = sha1(ts_full.encode("utf-8")).hexdigest()[:6]
    return f"{date_part}-{commit_part}-{ts_part}"


if __name__ == "__main__":

    run_tag = _build_run_tag(datetime.now())
    guardrail_runtime = JobGuardrails.build_runtime_context()
    guardrails = JobGuardrails.from_config(runtime_context=guardrail_runtime)

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
    log.info(f"Iniciando job run_tag={run_tag}.")
    if guardrails.is_enabled():
        snapshot = guardrails.snapshot()
        log.info(
            guardrails.format_snapshot(
                snapshot,
                context={"event": "job_start"},
            )
        )

    if Config.get_nested("postprocessing", "build_target") & Config.get_nested(
        "solver", "multiobjective"
    ):
        log.error("Build target not supported for multiobjective.")
        raise Exception("Build target not supported for multiobjective.")

    def _should_include_prp_file(file_name):
        if not file_name.startswith("PRP"):
            return True

        instance_part = file_name[3:].split("_", 1)[0]
        if not instance_part.isdigit():
            return True

        return int(instance_part) < 31

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
            if not _should_include_prp_file(file):
                continue

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

    worker_process = WorkerProcess(
        workers,
        timeSupervisor,
        {"instancia": log, "dirLogs": f"{output}logs"},
        guardrail_runtime=guardrail_runtime,
        run_tag=run_tag,
    )
    worker_process.run_parallel(instancies=instancies, solver=method)

    postprocessing = PostProcessingProcess(log=log, output=output)
    build_target = Config.get_nested("postprocessing", "build_target")
    union_path = postprocessing.union_results(
        run_tag=run_tag, build_target=build_target
    )
    if build_target:
        postprocessing.build_target(union_results_path=union_path)

    if guardrails.is_enabled():
        snapshot = guardrails.snapshot()
        log.info(
            guardrails.format_snapshot(
                snapshot,
                context={"event": "job_end"},
            )
        )

    """
Explored 11164 nodes (448772 simplex iterations) in 30.82 seconds (21.64 work units)
Thread count was 1 (of 8 available processors)

Solution count 10: 200744 200918 201025 ... 205818

Explored 8394 nodes (406720 simplex iterations) in 27.97 seconds (19.82 work units)
Thread count was 1 (of 8 available processors)

Solution count 10: 200744 201085 201197 ... 203144


"""
