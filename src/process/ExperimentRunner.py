"""Run the experiments listed in ``config.experiments`` one after another in a single job.

Each experiment is the base configuration plus its overrides (``Config.experiments``)
and writes under ``<instance.output>/<name>/``. An experiment finishes completely,
including its ``union_results``, before the next one starts, so a job submitted to
the cluster queue can evaluate several configurations.

Every experiment folder keeps an ``experiment_state.json`` that records the exact
``union_results`` file written, the configuration fingerprint and the expected
number of tasks. A resubmission skips an experiment only after opening that file and
checking the experiment name, the fingerprint and the task count.
"""

import json
import os
import shutil
import time
import traceback
from hashlib import sha256

import pandas as pd

from config import Config
from src.process.PostProcessingProcess import PostProcessingProcess
from src.process.RuntimeContext import (
    build_instances,
    build_runtime_context,
    create_run_logger,
    finalize_postprocessing,
    list_instance_files,
    validate_runtime_context,
)
from src.process.WorkerProcess import WorkerProcess

STATE_FILE = "experiment_state.json"

RUN = "run"
SKIP = "skip"
RESTART_CLEAN = "restart_clean"


def _say(message):
    print(f"[experimentos] {message}", flush=True)


def experiment_fingerprint(config):
    """Stable hash of an experiment configuration (workers and logging excluded)."""
    payload = {k: v for k, v in config.items() if k not in ("workers", "logging")}
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return sha256(text.encode("utf-8")).hexdigest()


def state_path(output):
    return os.path.join(output, STATE_FILE)


def read_state(output):
    path = state_path(output)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            state = json.load(handle)
    except (OSError, ValueError):
        return None
    return state if isinstance(state, dict) else None


def write_state(output, state):
    os.makedirs(output, exist_ok=True)
    with open(state_path(output), "w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2, sort_keys=True)


def _has_results(output):
    if not os.path.isdir(output):
        return False
    return any(name != STATE_FILE for name in os.listdir(output))


def validate_union(state, name, fingerprint, expected_tasks):
    """Open the union file recorded in ``state`` and check it belongs to ``name``.

    Returns ``(ok, reason)``. The file is the exact path stored in the state, never
    a glob of the experiment folder.
    """
    path = state.get("union_path")
    if not path or not os.path.isfile(path):
        return False, f"union_results ausente ({path})"
    try:
        frame = pd.read_excel(
            path,
            usecols=["experiment", "experiment_fingerprint", "task_key"],
            engine="openpyxl",
        )
    except Exception as error:  # unreadable or without the key columns
        return False, f"union_results ilegível ou sem chaves ({error})"
    if set(frame["experiment"].dropna().astype(str)) != {name}:
        return False, "union_results pertence a outro experimento"
    if set(frame["experiment_fingerprint"].dropna().astype(str)) != {fingerprint}:
        return False, "union_results de outra configuração (fingerprint diferente)"
    found = frame["task_key"].dropna().nunique()
    if found != expected_tasks:
        return False, f"union_results com {found} tarefas, esperado {expected_tasks}"
    return True, ""


def decide_action(output, name, fingerprint, expected_tasks):
    """``(action, reason)`` with action ``run``, ``skip`` or ``restart_clean``."""
    state = read_state(output)
    if state is None:
        if _has_results(output):
            return RESTART_CLEAN, "pasta com resultados sem estado do experimento"
        return RUN, "primeira execução"
    if state.get("experiment") != name or state.get("fingerprint") != fingerprint:
        return RESTART_CLEAN, "configuração do experimento mudou"
    if state.get("status") != "complete":
        return RUN, f"execução anterior {state.get('status', 'sem status')}"
    ok, reason = validate_union(state, name, fingerprint, expected_tasks)
    if ok:
        return SKIP, "experimento já concluído"
    return RUN, reason


def move_aside(output):
    """Rename an experiment folder to ``<name>.stale-<timestamp>``; nothing is deleted."""
    source = output.rstrip("/\\")
    target = f"{source}.stale-{time.strftime('%Y%m%d-%H%M%S')}"
    shutil.move(source, target)
    return target


def expected_task_count(worker):
    files = list_instance_files(
        Config.get_nested("instance", "dir"), Config.get_nested("instance", "files")
    )
    return worker.count_tasks(files)


def run_experiment(name, config, executor, managed=True):
    """Run one experiment already installed in ``Config``; returns its final state."""
    fingerprint = experiment_fingerprint(config)
    context = build_runtime_context()
    worker = WorkerProcess(
        context.workers,
        None,
        run_tag=context.run_tag,
        commit_hash=context.commit_hash,
        config_hash=context.config_hash,
        config_json=context.config_json,
    )
    expected = expected_task_count(worker)

    if managed:
        action, reason = decide_action(context.output, name, fingerprint, expected)
        if action == SKIP:
            _say(f"{name}: pulado ({reason}).")
            return {"experiment": name, "status": "skipped"}
        if action == RESTART_CLEAN:
            moved = move_aside(context.output)
            _say(f"{name}: {reason}; pasta anterior movida para {moved}.")
        else:
            _say(f"{name}: executando ({reason}).")

    log = create_run_logger(context.output)
    worker.log = log
    log.info(f"Iniciando experimento {name} run_tag={context.run_tag}.")
    validate_runtime_context(context, log)
    state = {
        "experiment": name,
        "fingerprint": fingerprint,
        "status": "running",
        "run_tag": context.run_tag,
        "expected_tasks": expected,
        "union_path": None,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if managed:
        write_state(context.output, state)

    instances = build_instances(
        context.instance_dir,
        context.output,
        context.files,
        context.threads_limit,
        context.time_limit,
    )
    worker.run_parallel(instancies=instances, solver=context.method, executor=executor)

    postprocessing = PostProcessingProcess(log=log, output=context.output)
    union_path = finalize_postprocessing(
        postprocessing,
        context.run_tag,
        context.build_target,
        experiment=name,
        fingerprint=fingerprint,
    )
    state.update(
        union_path=os.path.abspath(union_path) if union_path else None,
        finished_at=time.strftime("%Y-%m-%d %H:%M:%S"),
    )
    found = 0
    if union_path and os.path.isfile(union_path):
        keys = pd.read_excel(union_path, usecols=["task_key"], engine="openpyxl")
        found = int(keys["task_key"].dropna().nunique())
    state["found_tasks"] = found
    state["status"] = "complete" if found == expected else "incomplete"
    if state["status"] == "incomplete":
        log.error(
            f"Experimento {name} incompleto: {found} de {expected} tarefas no union_results."
        )
    if managed:
        write_state(context.output, state)
    _say(f"{name}: {state['status']} ({found}/{expected} tarefas).")
    return state


def _max_tasks(experiments):
    largest = 0
    for _, config in experiments:
        Config.replace(config)
        worker = WorkerProcess(Config.get_nested("workers", "num"), None)
        largest = max(largest, expected_task_count(worker))
    return largest


def run_experiments():
    """Run every configured experiment; returns the names of those that failed."""
    experiments = Config.experiments()  # validates every experiment up front
    base = Config.snapshot()
    managed = bool(base.get("experiments"))
    base_workers = Config.get_nested("workers", "num")
    failed = []
    executor = None
    try:
        largest = _max_tasks(experiments)
        executor = WorkerProcess(base_workers, None).create_executor(largest)
        for name, config in experiments:
            Config.replace(config)
            try:
                state = run_experiment(name, config, executor, managed=managed)
                if state["status"] == "incomplete":
                    failed.append(name)
            except Exception:
                failed.append(name)
                _say(f"{name}: falhou\n{traceback.format_exc()}")
    finally:
        if executor is not None:
            executor.shutdown(wait=True)
        Config.replace(base)  # leave the process with the base configuration
    _say(
        "concluído"
        + (f"; com falha ou incompletos: {', '.join(failed)}" if failed else "")
        + "."
    )
    return failed
