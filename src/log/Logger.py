import os
import sys
from datetime import datetime

from config import Config


class Logger:
    _LEVELS = {"DEBUG", "INFO", "OFF"}

    def __init__(
        self, log_dir="logs", log_file="app.log", worker_id=None, task=None, level=None
    ):

        os.makedirs(log_dir, exist_ok=True)
        self.log_path = os.path.join(log_dir, log_file)
        self.worker_id = worker_id
        self.task = task
        configured_level = level or Config.get_nested("logging", "level", default="INFO")
        self.level = self._normalize_level(configured_level)

    @classmethod
    def _normalize_level(cls, level):
        normalized = str(level or "INFO").upper()
        return normalized if normalized in cls._LEVELS else "INFO"

    def _timestamp(self):
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _should_log(self, level):
        if level in {"WARNING", "ERROR"}:
            return True
        if self.level == "DEBUG":
            return level in {"DEBUG", "INFO"}
        return self.level == "INFO" and level == "INFO"

    def _write(self, level, message):
        if not self._should_log(level):
            return

        level_label = f"{level} [Worker {self.worker_id}]"
        log_message = f"[{self._timestamp()}] [{level_label}] {message}\n"

        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(log_message)
            f.flush()

        print(log_message, end="", flush=True)
        sys.stdout.flush()

    def info(self, message):
        self._write("INFO", message)

    def warning(self, message):
        self._write("WARNING", message)

    def error(self, message):
        self._write("ERROR", message)

    def debug(self, message):
        self._write("DEBUG", message)

    def log(self, level, message):
        self._write(str(level).upper(), message)
