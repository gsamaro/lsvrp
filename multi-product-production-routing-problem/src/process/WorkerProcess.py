import os
import queue
import threading
import time
import traceback
from concurrent.futures import as_completed
from multiprocessing import Pool
from math import ceil

from config import Config
from src.helpers.JobGuardrails import JobGuardrails
from src.helpers.TargetsLoader import load_targets_by_file
from src.log.Logger import Logger
from src.process.InstanceProcess import InstanceProcess

try:
    from mpi4py import MPI
    from mpi4py.futures import MPIPoolExecutor

    MPI_BOOL = True
except:
    print("mpi4py not running")
    MPI_BOOL = False

from constants import WEIGHTS_OPTIMIZE, WEIGHTS_TARGET, ALPHA

if Config.get_nested("postprocessing", "build_target"):
    WEIGHTS = WEIGHTS_TARGET
else:
    WEIGHTS = WEIGHTS_OPTIMIZE


class WorkerProcess:

    def __init__(
        self, numWorkers=1, timeSupervisor=1, log: Logger = None, guardrail_runtime=None
    ):
        self.taskQueue = queue.Queue()
        self.numWorkers = numWorkers
        self.timeSupervisor = timeSupervisor
        self.dirLogs = log["dirLogs"]
        self.log: Logger = log["instancia"]
        self.targets_by_file = None
        self.guardrail_runtime = guardrail_runtime or JobGuardrails.build_runtime_context()
        self.guardrails = JobGuardrails.from_config(runtime_context=self.guardrail_runtime)
        self.preventive_stop = False
        self.mpi_batch_multiplier = max(
            1, int(Config.get_nested("workers", "mpi_batch_multiplier", default=1) or 1)
        )
        self.mpi_batch_max_tasks = Config.get_nested(
            "workers", "mpi_batch_max_tasks", default=None
        )
        if self.mpi_batch_max_tasks is not None:
            self.mpi_batch_max_tasks = max(1, int(self.mpi_batch_max_tasks))
        self.mpi_submit_guardrail_enabled = bool(
            Config.get_nested("workers", "mpi_submit_guardrail_enabled", default=True)
        )
        self.mpi_heavy_instance_cap = Config.get_nested(
            "workers", "mpi_heavy_instance_cap", default=None
        )
        if self.mpi_heavy_instance_cap is not None:
            self.mpi_heavy_instance_cap = max(1, int(self.mpi_heavy_instance_cap))

    def _ensure_targets_loaded(self):
        if self.targets_by_file is not None:
            return

        post_out = Config.get_nested("postprocessing", "output")
        if not post_out:
            self.targets_by_file = {}
            return

        targets_path = os.path.join(post_out, "targets.xlsx")
        self.targets_by_file = load_targets_by_file(targets_path, log=self.log)

    def _available_mpi_workers(self):
        pbs_np = os.environ.get("PBS_NP")
        if pbs_np:
            return int(pbs_np)

        pbs_nodefile = os.environ.get("PBS_NODEFILE")
        if pbs_nodefile and os.path.exists(pbs_nodefile):
            with open(pbs_nodefile, "r", encoding="utf-8") as f:
                return sum(1 for line in f if line.strip())

        for env_name in ("SLURM_NTASKS", "OMPI_COMM_WORLD_SIZE", "PMI_SIZE"):
            value = os.environ.get(env_name)
            if value:
                return int(value)

        return None

    def _resolve_num_workers(self, total_tasks):
        configured = self.numWorkers
        if isinstance(configured, str):
            configured = configured.strip().lower()

        if configured in ("auto", "max", "all", 0, "0", None):
            available = self._available_mpi_workers()
            if available is None:
                available = os.cpu_count() or 1
            return max(1, min(int(available), total_tasks))

        return max(1, min(int(configured), total_tasks))

    def _format_weight(self, weight):
        if weight is None:
            return "[]"

        formatted = []
        for value in weight:
            if isinstance(value, float):
                formatted.append(f"{value:g}")
            else:
                formatted.append(str(value))
        return "[" + ", ".join(formatted) + "]"

    def _log_guardrail(self, event, force=False, **context):
        if not self.guardrails.is_enabled():
            return None

        if not self.guardrails.should_emit_log(force=force) and not force:
            return None

        snapshot = self.guardrails.snapshot()
        message = self.guardrails.format_snapshot(snapshot, context={"event": event, **context})
        if snapshot.should_stop:
            self.log.error(message)
        elif snapshot.should_warn:
            self.log.warning(message)
        else:
            self.log.info(message)
        return snapshot

    def _stop_before_new_batch(self, mpi_batch, completed, total_tasks):
        snapshot = self._log_guardrail(
            "pre_batch_check",
            force=True,
            mpi_batch=mpi_batch,
            completed=completed,
            total=total_tasks,
        )
        if snapshot and snapshot.should_stop:
            self.preventive_stop = True
            self.log.warning(
                f"Guardrail preventivo acionado antes do lote MPI {mpi_batch}. Nenhum novo lote sera iniciado."
            )
            return True
        return False

    def _build_task_context(self, instancie, weight, alpha, mpi_batch=1, task_number=None):
        context = {
            "file": instancie["file"],
            "weight": self._format_weight(weight),
            "alpha": alpha,
            "mpi_batch": mpi_batch,
        }
        if task_number is not None:
            context["task_number"] = task_number

        context["label"] = (
            f"arquivo: {context['file']} | "
            f"peso: {context['weight']} | "
            f"alpha: {context['alpha']} | "
            f"lote MPI: {context['mpi_batch']}"
        )
        return context

    def _is_heavy_instance_file(self, instance_file):
        normalized = os.path.normpath(instance_file)
        path_parts = normalized.split(os.sep)
        return "DATA_PRP_30C" in path_parts

    def _batch_contains_heavy_instances(self, batch):
        return any(self._is_heavy_instance_file(task["instancie"]["file"]) for task in batch)

    def _effective_workers_for_tasks(self, total_tasks):
        return self._resolve_num_workers(total_tasks)

    def _compute_mpi_batch_size(self, total_tasks):
        effective_workers = self._effective_workers_for_tasks(total_tasks)
        batch_size = self.mpi_batch_multiplier * effective_workers
        if self.mpi_batch_max_tasks is not None:
            batch_size = min(batch_size, self.mpi_batch_max_tasks)
        batch_size = min(total_tasks, batch_size)
        return max(1, batch_size), effective_workers

    def _compute_batch_execution_plan(self, batch, total_tasks):
        effective_workers = self._effective_workers_for_tasks(total_tasks)
        batch_size = len(batch)
        heavy_batch = self._batch_contains_heavy_instances(batch)
        heavy_instance_cap = self.mpi_heavy_instance_cap if heavy_batch else None
        executor_workers = effective_workers
        if heavy_instance_cap is not None:
            executor_workers = min(executor_workers, heavy_instance_cap)
        executor_workers = max(1, min(executor_workers, batch_size))
        submission_window = batch_size
        if self.mpi_submit_guardrail_enabled:
            submission_window = executor_workers

        return {
            "effective_workers": effective_workers,
            "batch_size": batch_size,
            "executor_workers": executor_workers,
            "heavy_batch": heavy_batch,
            "heavy_instance_cap": heavy_instance_cap,
            "submission_window": submission_window,
        }

    def _iter_batches(self, tasks, batch_size):
        for start in range(0, len(tasks), batch_size):
            yield start, tasks[start : start + batch_size]

    def _log_mpi_batch_plan(
        self,
        mpi_batch,
        total_batches,
        batch,
        total_tasks,
        completed_tasks,
        failed_tasks,
        plan,
    ):
        pending = total_tasks - completed_tasks - failed_tasks
        self.log.info(
            "MPI_BATCH_PLAN "
            f"batch={mpi_batch}/{total_batches} "
            f"effective_workers={plan['effective_workers']} "
            f"batch_size={plan['batch_size']} "
            f"executor_workers={plan['executor_workers']} "
            f"submitted_tasks={len(batch)} "
            f"remaining_tasks={pending} "
            f"total_tasks={total_tasks} "
            f"heavy_batch={str(plan['heavy_batch']).lower()} "
            f"heavy_instance_cap={plan['heavy_instance_cap']} "
            f"submission_window={plan['submission_window']}"
        )

    def _submit_mpi_batch(self, executor, batch, mpi_batch, plan, solver):
        future_contexts = {}
        submission_window = max(1, plan["submission_window"])
        for start in range(0, len(batch), submission_window):
            window = batch[start : start + submission_window]
            for task in window:
                context = self._build_task_context(
                    task["instancie"],
                    task["weight"],
                    task["alpha"],
                    mpi_batch=mpi_batch,
                    task_number=task["task_number"],
                )
                future = executor.submit(
                    process,
                    task["log"],
                    task["instancie"],
                    solver,
                    task["weight"],
                    task["targets_by_file"],
                    task["alpha"],
                    context,
                    self.guardrail_runtime,
                )
                future_contexts[future] = context
        return future_contexts

    def supervisor(self):
        while True:
            time.sleep(self.timeSupervisor)
            size = self.taskQueue.qsize()
            self.log.info(f"[Supervisor] Tarefas restantes: {size}")
            if size == 0:
                self.log.info("[Supervisor] Fila vazia. Aguardando workers terminarem.")
                break

    def worker(self, worker_id, solver):
        self._ensure_targets_loaded()
        while True:
            try:
                task = self.taskQueue.get(timeout=2)
                log = Logger(
                    log_dir=self.dirLogs,
                    log_file=f"Worker_{worker_id}.log",
                    worker_id=worker_id,
                    task=task,
                )

            except queue.Empty:
                log.info(f" Fila vazia, encerrando.")
                break

            log.info(f" Processando nova Instância ")

            try:
                InstanceProcess(
                    task["instancie"]["file"],
                    task["instancie"]["output"],
                    isPloat=task["instancie"]["isPloat"],
                    timeLimit=task["instancie"]["timeLimit"],
                    numThreads=task["instancie"]["numThreads"],
                    log=log,
                    solver=solver,
                    targets_by_file=self.targets_by_file,
                ).process()

            except Exception as e:
                log.error(
                    f"Ocorreu um erro inesperado: {e}: stack: {traceback.format_exc()}"
                )

            log.info(f"Instância Finalizada")

            self.taskQueue.task_done()

    def old_process(self, instancies=[], solver="DEFAULT"):

        for i in range(len(instancies)):
            self.taskQueue.put({"task": {i + 1}, "instancie": instancies[i]})

        workers = []
        for i in range(self.numWorkers):
            self.worker(i + 1, solver)
            # t = threading.Thread(target=self.worker, args=(i+1, solver))
            # t.start()
            # workers.append(t)

        # supervisor_thread = threading.Thread(target=self.supervisor)
        # supervisor_thread.start()

        # for t in workers:
        #     t.join()

        # supervisor_thread.join()

        self.log.info(">> Fim do processamento.")

    def run_parallel(self, instancies=[], solver="GUROBY"):
        self.log.info(">> Iniciando processamento paralelo.")
        self._ensure_targets_loaded()
        tasks = []
        task_number = 1
        for instancie in instancies:
            for weight in WEIGHTS:
                for alpha in ALPHA:
                    tasks.append(
                        {
                            "log": self.log,
                            "instancie": instancie,
                            "weight": weight,
                            "targets_by_file": self.targets_by_file,
                            "alpha": alpha,
                            "task_number": task_number,
                        }
                    )
                    task_number += 1

        total_tasks = len(tasks)
        completed_tasks = 0
        failed_tasks = 0
        self._log_guardrail("run_parallel_start", force=True, completed=0, total=total_tasks)

        if MPI_BOOL:
            self.log.info(">> Iniciando processamento paralelo com MPI.")
            batch_size, effective_workers = self._compute_mpi_batch_size(total_tasks)
            self.log.info(
                f">> Workers MPI disponíveis: {effective_workers}. Tamanho planejado do lote: {batch_size}."
            )
            total_batches = ceil(len(tasks) / batch_size)
            for start, batch in self._iter_batches(tasks, batch_size):
                mpi_batch = start // batch_size + 1
                if self._stop_before_new_batch(mpi_batch, completed_tasks, total_tasks):
                    break
                plan = self._compute_batch_execution_plan(batch, total_tasks)
                self._log_mpi_batch_plan(
                    mpi_batch,
                    total_batches,
                    batch,
                    total_tasks,
                    completed_tasks,
                    failed_tasks,
                    plan,
                )
                self.log.info(
                    f"MPI_BATCH_START batch={mpi_batch}/{total_batches} tasks={len(batch)} completed={completed_tasks} failed={failed_tasks} pending={total_tasks - completed_tasks - failed_tasks}"
                )
                pending = total_tasks - completed_tasks - failed_tasks
                try:
                    with MPIPoolExecutor(max_workers=plan["executor_workers"]) as executor:
                        future_contexts = self._submit_mpi_batch(
                            executor,
                            batch,
                            mpi_batch,
                            plan,
                            solver,
                        )

                        for future in as_completed(future_contexts):
                            try:
                                future.result()
                                completed_tasks += 1
                            except Exception as e:
                                failed_tasks += 1
                                context = future_contexts.get(future)
                                context_label = (
                                    context["label"] if context else "contexto indisponivel"
                                )
                                self.log.error(
                                    f"Erro em tarefa MPI ({context_label}): {e}: stack: {traceback.format_exc()}"
                                )
                            finally:
                                pending = total_tasks - completed_tasks - failed_tasks
                                self._log_guardrail(
                                    "task_completion",
                                    mpi_batch=mpi_batch,
                                    completed=completed_tasks,
                                    total=total_tasks,
                                )
                except Exception as e:
                    pending = total_tasks - completed_tasks - failed_tasks
                    self.log.error(
                        f"guardrail event=mpi_batch_abort batch={mpi_batch}/{total_batches} submitted={len(batch)} completed={completed_tasks} failed={failed_tasks} pending={pending} error={e}"
                    )
                    raise

                self.log.info(
                    f"MPI_BATCH_END batch={mpi_batch}/{total_batches} completed={completed_tasks} failed={failed_tasks} pending={pending}"
                )
                snapshot = self._log_guardrail(
                    "post_batch_check",
                    force=True,
                    mpi_batch=mpi_batch,
                    completed=completed_tasks,
                    total=total_tasks,
                )
                if snapshot and snapshot.should_stop:
                    self.preventive_stop = True
                    self.log.warning(
                        f"Guardrail preventivo acionado apos o lote MPI {mpi_batch}. Nenhum novo lote sera iniciado."
                    )
                    break
        else:
            for task in tasks:
                if self._stop_before_new_batch(1, completed_tasks, total_tasks):
                    break
                context = self._build_task_context(
                    task["instancie"],
                    task["weight"],
                    task["alpha"],
                    mpi_batch=1,
                    task_number=task["task_number"],
                )
                process(
                    task["log"],
                    task["instancie"],
                    solver,
                    task["weight"],
                    task["targets_by_file"],
                    task["alpha"],
                    context,
                    self.guardrail_runtime,
                )
                completed_tasks += 1
                self._log_guardrail(
                    "task_completion",
                    completed=completed_tasks,
                    total=total_tasks,
                )
        self._log_guardrail(
            "run_parallel_end",
            force=True,
            completed=completed_tasks,
            total=total_tasks,
        )
        if self.preventive_stop:
            self.log.warning(
                f"Processamento encerrado preventivamente. completed={completed_tasks} failed={failed_tasks} total={total_tasks}"
            )
        self.log.info(">> Fim do processamento paralelo.")


def process(
    log,
    instancie,
    solver,
    w,
    targets_by_file,
    alpha,
    context=None,
    guardrail_runtime=None,
):
    context_label = context["label"] if context else instancie["file"]
    guardrails = JobGuardrails.from_config(runtime_context=guardrail_runtime)
    guardrails.refresh_runtime_context()
    current_runtime = guardrails.runtime_context()
    if guardrails.is_enabled():
        start_snapshot = guardrails.snapshot()
        log.info(
            guardrails.format_snapshot(
                start_snapshot,
                context={
                    "event": "instance_start",
                    "mpi_batch": context.get("mpi_batch") if context else None,
                    "task_number": context.get("task_number") if context else None,
                    "instance_file": instancie["file"],
                },
            )
        )
    log.info(f">> Processando instância ({context_label}).")
    # try:
    InstanceProcess(
        instancie["file"],
        instancie["output"],
        isPloat=False,
        timeLimit=instancie["timeLimit"],
        numThreads=instancie["numThreads"],
        log=log,
        solver=solver,
        weight=w,
        targets_by_file=targets_by_file,
        alpha=alpha,
        guardrail_runtime=current_runtime,
        task_context=context,
    ).process()
    # except Exception as e:
    #     log.error(
    #         f"Erro ao processar instância ({context_label}): {e}: stack: {traceback.format_exc()}"
    #     )
    #     raise RuntimeError(f"Falha na execução da instância ({context_label})") from e
    # finally:
    #     if guardrails.is_enabled():
    #         end_snapshot = guardrails.snapshot()
    #         log.info(
    #             guardrails.format_snapshot(
    #                 end_snapshot,
    #                 context={
    #                     "event": "instance_end",
    #                     "mpi_batch": context.get("mpi_batch") if context else None,
    #                     "task_number": context.get("task_number") if context else None,
    #                     "instance_file": instancie["file"],
    #                 },
    #             )
            # )
