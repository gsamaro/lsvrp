import threading
import queue
import time
from src.Logger import Logger
from src.InstanceProcess import InstanceProcess

class WorkerProcess:

    def __init__(self, numWorkers = 1, timeSupervisor = 1, dirLogs = "logs"):
        self.taskQueue = queue.Queue()
        self.numWorkers = numWorkers
        self.timeSupervisor = timeSupervisor
        self.dirLogs = dirLogs
        self.log = Logger(log_dir=dirLogs, log_file=f"Worker_0.log")

    def supervisor(self):
        while True:
            time.sleep(self.timeSupervisor)
            size = self.taskQueue.qsize()
            self.log.info(f"[Supervisor] Tarefas restantes: {size}")
            if size == 0:
                self.log.info("[Supervisor] Fila vazia. Aguardando workers terminarem.")
                break

    def worker(self,worker_id):
        log = Logger(log_dir=self.dirLogs, log_file=f"Worker_{worker_id}.log")

        while True:
            try:
                task = self.taskQueue.get(timeout=2)
            except queue.Empty:
                log.info(f"[Worker {worker_id}] Fila vazia, encerrando.")
                break

            log.info(f"[Worker {worker_id}] Processando: {task['task']}")
            
            InstanceProcess(
                task['instancie']['file'],
                task['instancie']['output'],
                isPloat=task['instancie']['isPloat'],
                timeLimit=task['instancie']['timeLimit'],
                numThreads=task['instancie']['numThreads']).process()

            log.info(f"[Worker {worker_id}] Finalizou: {task}")
            
            self.taskQueue.task_done()

    def process(self, instancies=[]):

        for i in range(len(instancies)):
            self.taskQueue.put({'task': {i+1},'instancie':instancies[i]})

        workers = []
        for i in range(self.numWorkers):
            t = threading.Thread(target=self.worker, args=(i+1,))
            t.start()
            workers.append(t)

        supervisor_thread = threading.Thread(target=self.supervisor)
        supervisor_thread.start()

        for t in workers:
            t.join()

        supervisor_thread.join()

        self.log.info(">> Fim do processamento.")