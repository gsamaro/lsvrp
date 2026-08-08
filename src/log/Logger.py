import os
import sys
from datetime import datetime


class Logger:
    def __init__(self, log_dir="logs", log_file="app.log", worker_id=None, task=None):

        os.makedirs(log_dir, exist_ok=True)
        self.log_path = os.path.join(log_dir, log_file)

        if not os.path.exists(self.log_path):
            with open(self.log_path, "w") as f:
                f.write(f"[{self._timestamp()}] Logger iniciado\n")
        self.worker_id = worker_id
        self.task = task

    def _timestamp(self):
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _write(self, level, message, only_file=False):
        log_message = f"[{self._timestamp()}] [{level}] {message}\n"

        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(log_message)
            f.flush()

        if not only_file:
            print(log_message, end="", flush=True)
            sys.stdout.flush()

    def info(self, message):
        self._write(f" INFO [Worker {self.worker_id}]", message)

    def warning(self, message):
        self._write(f" WARNING [Worker {self.worker_id}]", message)

    def error(self, message):
        self._write(f" ERROR [Worker {self.worker_id}]", message)

    def debug(self, message):
        self._write(f" DEBUG [Worker {self.worker_id}]", message)

    def log(self, level, message):
        self._write(level.upper(), message)
