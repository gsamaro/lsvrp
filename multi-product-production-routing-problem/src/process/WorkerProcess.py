import threading
import queue
import time
from src.log.Logger import Logger
from src.process.InstanceProcess import InstanceProcess
import traceback

class WorkerProcess:

    def __init__(self, numWorkers = 1, timeSupervisor = 1, log:Logger = None):
        self.taskQueue = queue.Queue()
        self.numWorkers = numWorkers
        self.timeSupervisor = timeSupervisor
        self.dirLogs = log['dirLogs']
        self.log:Logger = log['instancia'] 

    def supervisor(self):
        while True:
            time.sleep(self.timeSupervisor)
            size = self.taskQueue.qsize()
            self.log.info(f"[Supervisor] Tarefas restantes: {size}")
            if size == 0:
                self.log.info("[Supervisor] Fila vazia. Aguardando workers terminarem.")
                break

    def worker(self,worker_id,solver):
        while True:
            try:
                task = self.taskQueue.get(timeout=2)
                log = Logger(log_dir=self.dirLogs, log_file=f"Worker_{worker_id}.log", worker_id=worker_id, task = task)

            except queue.Empty:
                log.info(f" Fila vazia, encerrando.")
                break

            log.info(f" Processando nova Instância ")
            
            try:
                InstanceProcess(
                task['instancie']['file'],
                task['instancie']['output'],
                isPloat=task['instancie']['isPloat'],
                timeLimit=task['instancie']['timeLimit'],
                numThreads=task['instancie']['numThreads'],
                log=log,
                solver=solver).process()

            except Exception as e:
                log.error(f"Ocorreu um erro inesperado: {e}: stack: {traceback.format_exc()}")
            
            log.info(f"Instância Finalizada")
            
            self.taskQueue.task_done()

    def process(self, instancies=[], solver="DEFAULT"):

        for i in range(len(instancies)):
            self.taskQueue.put({'task': {i+1},'instancie':instancies[i]})

        workers = []
        for i in range(self.numWorkers):
            t = threading.Thread(target=self.worker, args=(i+1, solver))
            t.start()
            workers.append(t)

        supervisor_thread = threading.Thread(target=self.supervisor)
        supervisor_thread.start()

        for t in workers:
            t.join()

        supervisor_thread.join()

        self.log.info(">> Fim do processamento.")