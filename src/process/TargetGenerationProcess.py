"""Gera ``targets.xlsx`` (ideal, nadir e targets) a partir da relaxação linear.

A configuração vem do bloco ``target_generation`` de ``config.json``. Cada instância
é uma tarefa (``WorkerProcess.run_callables``, MPI ou sequencial) que resolve os LPs
de ``src.helpers.FaceNadirTargets`` e guarda o resultado em ``<output>/work/``; a
montagem do ``targets.xlsx`` só acontece quando todas as instâncias terminaram.

A pasta de saída guarda ``targets_state.json`` com a configuração resolvida, o
fingerprint, o commit, as instâncias, o número de tarefas esperado e o hash do
``targets.xlsx``. Uma pasta com targets sem estado, ou de outra configuração, é
movida para ``<pasta>.stale-<timestamp>``; nada é apagado. Com o mesmo fingerprint,
uma execução interrompida continua do que já está em ``work/``.
"""

import json
import os
import re
import subprocess
import time
import traceback
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd

from config import Config
from src.helpers.FaceNadirTargets import (
    BASE_COLUMNS,
    NADIR_FACE,
    TARGET_COLUMNS,
    compute_instance_lps,
    instance_targets,
    library_versions,
    target_rows,
    validate_target_rows,
)
from src.process.ExperimentRunner import experiment_fingerprint, move_aside
from src.process.RuntimeContext import create_run_logger, list_instance_files
from src.process.WorkerProcess import WorkerProcess

STATE_FILE = "targets_state.json"
TARGETS_FILE = "targets.xlsx"
WORK_DIR = "work"

RUN = "run"
SKIP = "skip"
RESTART_CLEAN = "restart_clean"

_REPOSITORY = Path(__file__).resolve().parents[2]


def _say(message):
    print(f"[targets] {message}", flush=True)


def generation_settings(config=Config):
    """Bloco ``target_generation`` normalizado, mais o diretório das instâncias."""
    settings = config.snapshot()["target_generation"]
    settings["instance"]["dir"] = config.get_nested("instance", "dir")
    return settings


def effective_config(settings):
    """Configuração completa usada pelas tarefas, derivada só do bloco.

    As demais seções de ``config.json`` não entram: o resultado depende apenas do
    bloco ``target_generation`` e dos defaults do código.
    """
    formulation = settings["formulation"]
    block = {key: value for key, value in settings.items() if key != "instance"}
    block["instance"] = {"files": settings["instance"]["files"]}
    return Config.normalize(
        {
            "solver": {
                "threadsLimit": 1,
                "multiobjective": False,
                "aggregate_vehicles": formulation["aggregate_vehicles"],
                "symmetric_routes": formulation["symmetric_routes"],
                "strengthened_bounds": formulation["strengthened_bounds"],
                "coelho_inequalities": formulation["coelho_inequalities"],
                "symmetry_breaking": {"hc1": formulation["symmetry_breaking_hc1"]},
            },
            "relaxed_solution": {
                "replace_model": True,
                "use": False,
                "target_use_relaxation": settings["relaxation"]["target_use_relaxation"],
            },
            "postprocessing": {"build_target": True, "output": settings["output"]},
            "instance": {
                "dir": settings["instance"]["dir"],
                "output": "./out/",
                "files": settings["instance"]["files"],
            },
            "target_generation": block,
        }
    )


def list_target_instances(settings):
    """Chaves ``file`` do ``targets.xlsx`` (``<dir><pasta>/<arquivo>.dat``), ordenadas."""
    instance_dir = settings["instance"]["dir"]
    listed = list_instance_files(instance_dir, settings["instance"]["files"])
    return sorted(f"{instance_dir}{data}/{name}" for data, name in listed)


def _git(*arguments):
    try:
        result = subprocess.run(
            ["git", *arguments],
            capture_output=True,
            text=True,
            check=True,
            cwd=_REPOSITORY,
        )
        return result.stdout.strip()
    except Exception:
        return None


def git_state():
    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain")
    return {"commit": commit or "desconhecido", "dirty": bool(status)}


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
    path = state_path(output)
    with open(path + ".tmp", "w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2, sort_keys=True)
    os.replace(path + ".tmp", path)


def file_sha256(path):
    digest = sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _has_content(output):
    return os.path.isdir(output) and any(os.scandir(output))


def cache_path(output, instance):
    name = re.sub(r"[^A-Za-z0-9_.-]", "_", instance.lstrip("./")) + ".json"
    return os.path.join(output, WORK_DIR, name)


def read_cache(output, instance, fingerprint):
    """Resultado guardado da instância, ou ``None`` se ausente ou de outra execução."""
    path = cache_path(output, instance)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError):
        return None
    if payload.get("instance") != instance or payload.get("fingerprint") != fingerprint:
        return None
    return payload


def _nan_to_none(array):
    return None if array is None else np.where(np.isnan(array), None, array).tolist()


def solve_instance_task(payload):
    """Tarefa de uma instância (executa no worker MPI); nunca propaga exceção."""
    started = time.time()
    try:
        Config.replace(json.loads(payload["config_json"]))
        result = compute_instance_lps(
            payload["instance"],
            payload["weights"],
            nadir_method=payload["nadir_method"],
            epsilon=payload["epsilon"],
            lp_algorithm=payload["lp_algorithm"],
        )
        record = {
            "instance": payload["instance"],
            "fingerprint": payload["fingerprint"],
            "periods": result["periods"],
            "z": result["z"].tolist(),
            "vertex": result["vertex"].tolist(),
            "face_minima": _nan_to_none(result["face_minima"]),
            "features": result["features"],
            "seconds": result["seconds"],
            "build_seconds": result["build_seconds"],
            "versions": library_versions(),
        }
        path = payload["cache_path"]
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path + ".tmp", "w", encoding="utf-8") as handle:
            json.dump(record, handle)
        os.replace(path + ".tmp", path)
        return {"instance": payload["instance"], "seconds": time.time() - started}
    except Exception as error:
        return {
            "instance": payload["instance"],
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
        }


def _array(values):
    return np.array(values, dtype=float)


def build_targets_frame(settings, instances, caches):
    """``targets.xlsx`` completo: só ``file``, ``time`` e as 15 colunas de valores.

    A proveniência (fingerprint, commit, versões) fica em ``targets_state.json``.
    """
    parts = []
    for instance in instances:
        cache = caches[instance]
        face = cache["face_minima"]
        ideal, nadir, target = instance_targets(
            _array(cache["vertex"]),
            None if face is None else _array(face),
            settings["nadir"]["method"],
            settings["factor"],
        )
        rows = target_rows(instance, ideal, nadir, target)
        validate_target_rows(rows, cache["periods"])
        parts.append(rows)
    frame = pd.concat(parts, ignore_index=True)
    return frame.sort_values(BASE_COLUMNS, kind="stable").reset_index(drop=True)


def validate_output(state, fingerprint, instances):
    """``(ok, motivo)``: o ``targets.xlsx`` registrado no estado está íntegro."""
    path = state.get("targets_path")
    if not path or not os.path.isfile(path):
        return False, f"targets.xlsx ausente ({path})"
    if file_sha256(path) != state.get("targets_sha256"):
        return False, "hash do targets.xlsx difere do estado"
    try:
        frame = pd.read_excel(path, usecols=["file", "time"], engine="openpyxl")
    except Exception as error:
        return False, f"targets.xlsx ilegível ({error})"
    if state.get("fingerprint") != fingerprint:
        return False, "targets.xlsx de outra configuração"
    if set(frame["file"]) != set(instances):
        return False, "instâncias do targets.xlsx diferem das esperadas"
    return True, ""


def decide_action(output, fingerprint, instances):
    """``(ação, motivo)`` com ação ``run``, ``skip`` ou ``restart_clean``."""
    state = read_state(output)
    if state is None:
        if _has_content(output):
            return RESTART_CLEAN, "pasta com arquivos sem estado de geração de targets"
        return RUN, "primeira execução"
    if state.get("fingerprint") != fingerprint:
        return RESTART_CLEAN, "configuração de geração mudou"
    if state.get("status") == "complete":
        ok, reason = validate_output(state, fingerprint, instances)
        if ok:
            return SKIP, "targets já gerados com esta configuração"
        return RUN, reason
    return RUN, f"execução anterior {state.get('status', 'sem status')}"


def build_state(effective, fingerprint, instances, provenance):
    return {
        "status": "running",
        "fingerprint": fingerprint,
        "config": effective,
        "commit": provenance["commit"],
        "dirty": provenance["dirty"],
        "started_at": provenance["generated_at"],
        "finished_at": None,
        "instances": instances,
        "expected_tasks": len(instances),
        "found_tasks": 0,
        "targets_path": None,
        "targets_sha256": None,
        "rows": 0,
        "versions": library_versions(),
    }


def plan_generation(config=Config):
    """Resumo do que a geração faria, sem criar nada."""
    settings = generation_settings(config)
    effective = effective_config(settings)
    fingerprint = experiment_fingerprint(effective)
    instances = list_target_instances(settings)
    output = settings["output"]
    action, reason = decide_action(output, fingerprint, instances)
    done = sum(
        1 for instance in instances if read_cache(output, instance, fingerprint)
    )
    return {
        "output": output,
        "fingerprint": fingerprint,
        "instances": len(instances),
        "already_computed": done if action == RUN else 0,
        "periods_lps_per_instance": (
            "5 + 20*T" if settings["nadir"]["method"] == NADIR_FACE else "5"
        ),
        "action": action,
        "reason": reason,
        "settings": settings,
    }


def run_target_generation(config=Config):
    """Gera os targets; devolve o estado final (``status`` ``complete`` ou ``incomplete``)."""
    base = config.snapshot()
    settings = generation_settings(config)
    effective = effective_config(settings)
    fingerprint = experiment_fingerprint(effective)
    instances = list_target_instances(settings)
    if not instances:
        raise ValueError("target_generation.instance.files não seleciona instâncias")
    output = settings["output"]

    action, reason = decide_action(output, fingerprint, instances)
    if action == SKIP:
        _say(f"pulado ({reason}).")
        return read_state(output)
    if action == RESTART_CLEAN:
        moved = move_aside(output)
        _say(f"{reason}; pasta anterior movida para {moved}.")
    else:
        _say(f"executando ({reason}).")

    os.makedirs(os.path.join(output, WORK_DIR), exist_ok=True)
    log = create_run_logger(output)
    provenance = git_state()
    provenance["generated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    state = build_state(effective, fingerprint, instances, provenance)
    write_state(output, state)
    log.info(
        f"Geração de targets fingerprint={fingerprint[:12]} "
        f"instâncias={len(instances)} commit={provenance['commit'][:12]}."
    )

    pending = [
        instance
        for instance in instances
        if read_cache(output, instance, fingerprint) is None
    ]
    log.info(f"{len(instances) - len(pending)} instâncias já calculadas, {len(pending)} a calcular.")
    pending.sort(key=lambda path: os.path.getsize(path), reverse=True)
    config_json = json.dumps(effective, sort_keys=True)
    payloads = [
        {
            "instance": instance,
            "fingerprint": fingerprint,
            "config_json": config_json,
            "weights": settings["weights"],
            "nadir_method": settings["nadir"]["method"],
            "epsilon": settings["nadir"]["epsilon"],
            "lp_algorithm": settings["lp"]["algorithm"],
            "cache_path": cache_path(output, instance),
        }
        for instance in pending
    ]
    failures = []
    try:
        worker = WorkerProcess(config.get_nested("workers", "num"), log)
        results = worker.run_callables(
            solve_instance_task, payloads, labels=pending
        )
        for index, result in results:
            if result.get("error"):
                failures.append(pending[index])
                log.error(
                    f"Falha em {pending[index]}: {result['error']}\n"
                    f"{result.get('traceback', '')}"
                )
    finally:
        config.replace(base)

    caches = {
        instance: read_cache(output, instance, fingerprint) for instance in instances
    }
    missing = [instance for instance, cache in caches.items() if cache is None]
    state["found_tasks"] = len(instances) - len(missing)
    if missing:
        state.update(status="incomplete", finished_at=time.strftime("%Y-%m-%d %H:%M:%S"))
        write_state(output, state)
        log.error(f"Geração incompleta: {len(missing)} instâncias sem resultado.")
        _say(f"incompleto ({state['found_tasks']}/{len(instances)} instâncias).")
        return state

    frame = build_targets_frame(settings, instances, caches)
    targets_path = os.path.join(output, TARGETS_FILE)
    temporary = os.path.join(output, "targets.partial.xlsx")
    frame.to_excel(temporary, index=False)
    os.replace(temporary, targets_path)
    state.update(
        status="complete",
        finished_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        targets_path=os.path.abspath(targets_path),
        targets_sha256=file_sha256(targets_path),
        rows=len(frame),
        versions={
            **library_versions(),
            "workers_cplex": sorted(
                {str(cache["versions"].get("cplex")) for cache in caches.values()}
            ),
        },
    )
    write_state(output, state)
    log.info(f"targets.xlsx salvo em {targets_path} ({len(frame)} linhas).")
    _say(f"completo ({state['found_tasks']}/{len(instances)} instâncias, {len(frame)} linhas).")
    return state
