import os
from datetime import datetime

class Logger:
    def __init__(self, log_dir="logs", log_file="app.log", worker_id=None, task = None):

        os.makedirs(log_dir, exist_ok=True) 
        self.log_path = os.path.join(log_dir, log_file)

        if not os.path.exists(self.log_path):
            with open(self.log_path, 'w') as f:
                f.write(f"[{self._timestamp()}] Logger iniciado\n")
        self.worker_id =worker_id
        self.task=task

    def _timestamp(self):
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _write(self, level, message):
        log_message = f"[{self._timestamp()}] [{level}] {message}\n"

        with open(self.log_path, 'a', encoding='utf-8') as f:
            print(log_message)
            f.write(log_message)

    def info(self, message):
        if(self.task == 0):
            self._write(f" INFO [Worker {self.worker_id}]", message)
        else: 
            self._write(f" INFO [Worker {self.worker_id}] Processando: {self.task['task']}, Arquivo: {self.task['instancie']['file']}", message)

    def warning(self, message):
        if(self.task == 0):
            self._write(f" WARNING [Worker {self.worker_id}]", message)
        else: 
            self._write(f" WARNING [Worker {self.worker_id}] Processando: {self.task['task']}, Arquivo: {self.task['instancie']['file']}", message)

    def error(self, message):
        if(self.task == 0):
            self._write(f" ERROR [Worker {self.worker_id}]", message)
        else: 
            self._write(f" ERROR [Worker {self.worker_id}] Processando: {self.task['task']}, Arquivo: {self.task['instancie']['file']}", message)
        
    def debug(self, message):
        if(self.task == 0):
            self._write(f" DEBUG [Worker {self.worker_id}]",message)
        else: 
            self._write(f" DEBUG [Worker {self.worker_id}] Processando: {self.task['task']}, Arquivo: {self.task['instancie']['file']}",message)

    def log(self, level, message):
        self._write(level.upper(), message)
