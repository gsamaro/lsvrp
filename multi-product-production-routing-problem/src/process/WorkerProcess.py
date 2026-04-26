import os
import queue
import threading
import time
import traceback
from multiprocessing import Pool

from config import Config
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

    def __init__(self, numWorkers=1, timeSupervisor=1, log: Logger = None):
        self.taskQueue = queue.Queue()
        self.numWorkers = numWorkers
        self.timeSupervisor = timeSupervisor
        self.dirLogs = log["dirLogs"]
        self.log: Logger = log["instancia"]
        self.targets_by_file = None

    def _ensure_targets_loaded(self):
        if self.targets_by_file is not None:
            return

        post_out = Config.get_nested("postprocessing", "output")
        if not post_out:
            self.targets_by_file = {}
            return

        targets_path = os.path.join(post_out, "targets.xlsx")
        self.targets_by_file = load_targets_by_file(targets_path, log=self.log)

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
        if MPI_BOOL:
            self.log.info(">> Iniciando processamento paralelo com MPI.")
            with MPIPoolExecutor() as executor:
                futures = executor.starmap(
                    process,
                    (
                        (self.log, i, w, self.targets_by_file, alpha)
                        for i in instancies
                        for w in WEIGHTS
                        for alpha in ALPHA
                    ),
                )
                executor.shutdown(wait=True)
        else:
            for i in instancies:
                for w in WEIGHTS:
                    for alpha in ALPHA: 
                        process(self.log, i, w, self.targets_by_file, alpha)
        self.log.info(">> Fim do processamento paralelo.")


def process(log, instancie, w, targets_by_file, alpha):
    log.info(">> Processando instância.")
    InstanceProcess(
        instancie["file"],
        instancie["output"],
        isPloat=False,
        timeLimit=instancie["timeLimit"],
        numThreads=instancie["numThreads"],
        log=log,
        solver="GUROBY",
        weight=w,
        targets_by_file=targets_by_file,
        alpha=alpha,
    ).process()
