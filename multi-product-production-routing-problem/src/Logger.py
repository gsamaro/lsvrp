import os
from datetime import datetime

class Logger:
    def __init__(self, log_dir="logs", log_file="app.log"):

        os.makedirs(log_dir, exist_ok=True) 
        self.log_path = os.path.join(log_dir, log_file)

        if not os.path.exists(self.log_path):
            with open(self.log_path, 'w') as f:
                f.write(f"[{self._timestamp()}] Logger iniciado\n")

    def _timestamp(self):
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _write(self, level, message):
        log_message = f"[{self._timestamp()}] [{level}] {message}\n"
        with open(self.log_path, 'a', encoding='utf-8') as f:
            f.write(log_message)

    def info(self, message):
        self._write("INFO", message)

    def warning(self, message):
        self._write("WARNING", message)

    def error(self, message):
        self._write("ERROR", message)

    def debug(self, message):
        self._write("DEBUG", message)

    def log(self, level, message):
        self._write(level.upper(), message)
