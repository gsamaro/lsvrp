import os
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime
from hashlib import sha1

from config import Config
from src.log.Logger import Logger


@dataclass(frozen=True)
class RuntimeContext:
    run_tag: str
    threads_limit: object
    time_limit: int
    workers: object
    output: str
    instance_dir: str
    files: object
    method: str
    build_target: bool
    multiobjective: bool


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
    ts_part = sha1(now.astimezone().isoformat().encode("utf-8")).hexdigest()[:6]
    return f"{date_part}-{commit_part}-{ts_part}"


def build_runtime_context(config=Config, now=None):
    threads_limit = config.get_nested("solver", "threadsLimit")
    if threads_limit == "None":
        threads_limit = None

    return RuntimeContext(
        run_tag=_build_run_tag(now or datetime.now()),
        threads_limit=threads_limit,
        time_limit=int(config.get_nested("solver", "timeLimit")),
        workers=config.get_nested("workers", "num"),
        output=config.get_nested("instance", "output"),
        instance_dir=config.get_nested("instance", "dir"),
        files=config.get_nested("instance", "files"),
        method=config.get_nested("solver", "method"),
        build_target=config.get_nested("postprocessing", "build_target"),
        multiobjective=config.get_nested("solver", "multiobjective"),
    )


def create_run_logger(output):
    log_dir = os.path.join(output, "logs")
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    return Logger(log_dir=log_dir, log_file="Worker_0.log", worker_id=0, task=0)


def validate_runtime_context(context, log):
    if context.build_target and context.multiobjective:
        log.error("Build target not supported for multiobjective.")
        raise RuntimeError("Build target not supported for multiobjective.")


def is_supported_prp_file(file_name):
    """Keep the experiment scope limited to PRP instances 1 through 30."""
    if not file_name.startswith("PRP"):
        return True

    instance_part = file_name[3:].split("_", 1)[0]
    if not instance_part.isdigit():
        return True

    return int(instance_part) < 31


def build_instances(instance_dir, output, files, num_threads, time_limit):
    instances = []
    for selection in files:
        if ".dat" in selection:
            data, file_name = selection.split("/", 1)
            candidates = [file_name]
        else:
            data = selection
            candidates = [
                file_name
                for file_name in os.listdir(instance_dir + data)
                if os.path.isfile(os.path.join(instance_dir + data, file_name))
            ]

        for file_name in candidates:
            if not is_supported_prp_file(file_name):
                continue

            output_file = f"{output}{data}/{file_name[:-4]}/"
            os.makedirs(output_file, exist_ok=True)
            instances.append(
                {
                    "file": f"{instance_dir}{data}/{file_name}",
                    "output": output_file,
                    "numThreads": num_threads,
                    "timeLimit": time_limit,
                }
            )
    return instances


def finalize_postprocessing(postprocessing, run_tag, build_target):
    union_path = postprocessing.union_results(
        run_tag=run_tag, build_target=build_target
    )
    if build_target:
        if not union_path:
            postprocessing.log.error(
                "Falha na consolidação; targets não serão gerados."
            )
            raise RuntimeError("Falha na consolidação antes da geração de targets.")
        postprocessing.build_target(union_results_path=union_path)
    return union_path
