import os
import traceback
from concurrent.futures import as_completed

from config import Config
from src.helpers.TargetsLoader import load_targets_by_file
from src.log.Logger import Logger

try:
    from mpi4py.futures import MPIPoolExecutor

    MPI_BOOL = True
except ImportError:
    MPI_BOOL = False

from constants import WEIGHTS_OPTIMIZE, WEIGHTS_TARGET, ALPHA

if Config.get_nested("postprocessing", "build_target"):
    WEIGHTS = WEIGHTS_TARGET
else:
    WEIGHTS = WEIGHTS_OPTIMIZE


class WorkerProcess:

    def __init__(self, numWorkers=1, log: Logger = None, run_tag=None):
        self.numWorkers = numWorkers
        self.log: Logger = log["instancia"]
        self.targets_by_file = None
        self.run_tag = run_tag

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

    def _build_task_context(self, instancie, weight, alpha, task_number=None):
        context = {
            "file": instancie["file"],
            "weight": self._format_weight(weight),
            "alpha": alpha,
            "run_tag": self.run_tag,
        }
        if task_number is not None:
            context["task_number"] = task_number

        context["label"] = (
            f"arquivo: {context['file']} | "
            f"peso: {context['weight']} | "
            f"alpha: {context['alpha']}"
        )
        return context

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

        if not tasks:
            self.log.info(">> Nenhuma tarefa para processar.")
            return

        if MPI_BOOL:
            max_workers = self._resolve_num_workers(len(tasks))
            self.log.info(
                f">> Iniciando processamento paralelo com MPI usando {max_workers} workers."
            )
            future_contexts = {}
            with MPIPoolExecutor(max_workers=max_workers) as executor:
                for task in tasks:
                    context = self._build_task_context(
                        task["instancie"],
                        task["weight"],
                        task["alpha"],
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
                    )
                    future_contexts[future] = context

                for future in as_completed(future_contexts):
                    try:
                        future.result()
                    except Exception as e:
                        context = future_contexts[future]
                        self.log.error(
                            f"Erro em tarefa MPI ({context['label']}): {e}: stack: {traceback.format_exc()}"
                        )
        else:
            self.log.warning("mpi4py indisponível; executando tarefas sequencialmente.")
            for task in tasks:
                context = self._build_task_context(
                    task["instancie"],
                    task["weight"],
                    task["alpha"],
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
                )
        self.log.info(">> Fim do processamento paralelo.")


def process(log, instancie, solver, w, targets_by_file, alpha, context=None):
    context_label = context["label"] if context else instancie["file"]
    log.debug(f">> Processando instância ({context_label}).")
    InstanceProcess(
        instancie["file"],
        instancie["output"],
        timeLimit=instancie["timeLimit"],
        numThreads=instancie["numThreads"],
        log=log,
        solver=solver,
        weight=w,
        targets_by_file=targets_by_file,
        alpha=alpha,
        task_context=context,
    ).process()
