import threading
import time

from src.helpers.JobGuardrails import JobGuardrails
from src.helpers.ReadPrpFile import ReadPrpFile as RD
from src.helpers.TargetsLoader import normalize_instance_file_key
from src.log.Logger import Logger
from src.process.ProcessResults import getResults
from src.solvers.MultProductProdctionRoutingProblem import (
    MultProductProdctionRoutingProblem as MPPRP,
)
from src.solvers.MultProductProdctionRoutingProblemGreedyConstructiveHeuristic import MultProductProdctionRoutingProblemGreedyConstructiveHeuristic as MPPRPG
import gc
import numpy as np


class InstanceProcess:

    def __init__(
        self,
        instance,
        output,
        isPloat="false",
        numThreads=None,
        timeLimit=None,
        log: Logger = None,
        solver="GUROBY",
        weight=None,
        targets_by_file=None,
        alpha=None,
        guardrail_runtime=None,
        task_context=None,
    ):
        self.instance = instance
        self.isPloat = isPloat
        self.numThreads = numThreads
        self.timeLimit = timeLimit
        self.output = output
        self.isFinished = False
        self.solver = solver
        self.log: Logger = log
        self.weight = weight
        self.targets_by_file = targets_by_file
        self.alpha = alpha
        self.guardrails = JobGuardrails.from_config(runtime_context=guardrail_runtime)
        self.guardrails.refresh_runtime_context()
        self.task_context = task_context or {}
        self._solve_watchdog_stop = None
        self._solve_watchdog_thread = None

    def isProcessFinished(self):
        return self.isFinished

    def _base_log_context(self):
        return {
            "instance_file": self.instance,
            "mpi_batch": self.task_context.get("mpi_batch"),
            "task_number": self.task_context.get("task_number"),
        }

    def _log_snapshot(self, event, level="info", **context):
        snapshot = self.guardrails.snapshot()
        method = getattr(self.log, level)
        method(
            self.guardrails.format_snapshot(
                snapshot,
                context={**self._base_log_context(), "event": event, **context},
            )
        )
        return snapshot

    def _log_memory_checkpoint(self, checkpoint, phase=None):
        if not self.guardrails.is_enabled() or not self.guardrails.log_memory_checkpoints:
            return

        self._log_snapshot(
            "memory_checkpoint",
            checkpoint=checkpoint,
            phase=phase,
        )

    def _log_instance_identity(self):
        if not self.guardrails.is_enabled() or not self.guardrails.log_rank_identity:
            return

        self._log_snapshot("instance_identity")

    def _log_phase(self, phase, started_at):
        if not self.guardrails.is_enabled() or not self.guardrails.log_solver_phase_timing:
            return

        self.log.info(
            self.guardrails.format_snapshot(
                self.guardrails.snapshot(),
                context={**self._base_log_context(), "event": f"phase_{phase}", "phase": phase},
            )
            + f" phase_elapsed={time.time() - started_at:.2f}s"
        )

    def _start_solve_watchdog(self):
        if (
            not self.guardrails.is_enabled()
            or not self.guardrails.solve_watchdog_enabled
        ):
            return

        interval = max(0.01, float(self.guardrails.solve_watchdog_interval_seconds))
        stop_event = threading.Event()
        self._solve_watchdog_stop = stop_event

        def _watchdog():
            seq = 0
            while not stop_event.wait(interval):
                seq += 1
                self._log_snapshot(
                    "solve_watchdog",
                    watchdog_seq=seq,
                    phase="solve",
                )

        self._solve_watchdog_thread = threading.Thread(
            target=_watchdog,
            name="solve-watchdog",
            daemon=True,
        )
        self._solve_watchdog_thread.start()

    def _stop_solve_watchdog(self):
        if self._solve_watchdog_stop is not None:
            self._solve_watchdog_stop.set()
        if self._solve_watchdog_thread is not None:
            self._solve_watchdog_thread.join(timeout=1)
        self._solve_watchdog_stop = None
        self._solve_watchdog_thread = None

    def solverInstancie(self, data):
        # Python 3.8-compatible replacement for match-case
        if self.solver == "GUROBY":
            self.log.info(f" Solver: GUROBY")
            return MPPRP(
                map=data, dir=self.output, log=self.log, start={"start": False}
            )
        elif self.solver == "HEURISTICA_CONSTRUTIVA":
            self.log.info(f"Solver: HEURISTICA_CONSTRUTIVA")
            inst = MPPRPG(
                map=data, dir=self.output, log=self.log, rng=np.random.default_rng(seed=123))
            inst.setMitStart(True)
            return inst
        else:
            print(" VALOR SETADO COMO DEFAULT ----- SEM SOLVER ")
            return 0

    def process(self):

        data = None
        instance = None
        results = None
        Z = X = Y = I = R = Q = P = None
        FO = GAP = TIME = EPSILON = SOL_COUNT = RELAXED_MODEL_OBJE_VAL = None
        NODE_COUNT = OBJ_BOUND = NEW_TARGETS = None

        try:
            self._log_instance_identity()
            self._log_memory_checkpoint("before_read_instance", phase="read_instance")
            phase_started_at = time.time()
            data = RD(file_path=self.instance, log=self.log).getDataSet()
            self._log_memory_checkpoint("after_read_instance", phase="read_instance")
            self._log_phase("read_instance", phase_started_at)
            data["weight"] = self.weight
            data["alpha"] = self.alpha

            targets = []
            if self.targets_by_file:
                key = normalize_instance_file_key(data.get("file"))
                targets = self.targets_by_file.get(key, [])
            data["targets"] = targets

            self._log_memory_checkpoint("before_build_solver", phase="build_solver")
            phase_started_at = time.time()
            instance = self.solverInstancie(data)
            self._log_memory_checkpoint("after_build_solver", phase="build_solver")
            self._log_phase("build_solver", phase_started_at)

            self._log_memory_checkpoint("before_solve", phase="solve")
            phase_started_at = time.time()
            self._start_solve_watchdog()
            instance.solver(timeLimit=self.timeLimit, numThreads=self.numThreads)
            self._stop_solve_watchdog()
            self._log_memory_checkpoint("after_solve", phase="solve")
            self._log_phase("solve", phase_started_at)

            self._log_memory_checkpoint("before_extract_results", phase="extract_results")
            phase_started_at = time.time()
            (
                Z,
                X,
                Y,
                I,
                R,
                Q,
                P,
                FO,
                GAP,
                TIME,
                EPSILON,
                SOL_COUNT,
                RELAXED_MODEL_OBJE_VAL,
                NODE_COUNT,
                OBJ_BOUND,
                NEW_TARGETS,
            ) = instance.getResults()
            self._log_memory_checkpoint("after_extract_results", phase="extract_results")
            self._log_phase("extract_results", phase_started_at)
            # FO, f1, f2, f3, f4, GAP, TIME, SOL_COUNT, RELAXED_MODEL_OBJE_VAL, NODE_COUNT, OBJ_BOUND = instance.new_get_results()

            self._log_memory_checkpoint("before_write_results", phase="write_results")
            phase_started_at = time.time()
            results = getResults(
                data,
                self.output,
                Z,
                X,
                Y,
                I,
                R,
                Q,
                P,
                FO,
                GAP,
                TIME,
                EPSILON,
                SOL_COUNT,
                RELAXED_MODEL_OBJE_VAL,
                NODE_COUNT,
                OBJ_BOUND,
                NEW_TARGETS,
                log=self.log,
                guardrails=self.guardrails,
                task_context=self._base_log_context(),
            )
            self._log_memory_checkpoint("after_write_results", phase="write_results")
            self._log_phase("write_results", phase_started_at)
            self.log.info(f"Resultados gerados.")
            # new_get_results(self.output, FO,f1,f2,f3,f4,GAP,TIME,SOL_COUNT,RELAXED_MODEL_OBJE_VAL,NODE_COUNT,OBJ_BOUND)

            # if(self.isPloat=='true'):
            #     graphResults(results['periods'],{'coordsX':data['coordXY']['x'],'coordsY':data['coordXY']['y']},self.output)

            self.isFinished = True
        finally:
            if instance is not None and hasattr(instance, "terminate"):
                try:
                    instance.terminate()
                except Exception as e:
                    if self.log:
                        self.log.error(f"Erro ao liberar modelo: {e}")

            self._stop_solve_watchdog()
            self._log_memory_checkpoint("before_gc_collect", phase="cleanup")
            del results
            del Z, X, Y, I, R, Q, P
            del FO, GAP, TIME, EPSILON, SOL_COUNT, RELAXED_MODEL_OBJE_VAL
            del NODE_COUNT, OBJ_BOUND, NEW_TARGETS
            del instance
            del data
            gc.collect()
            self._log_memory_checkpoint("after_gc_collect", phase="cleanup")
