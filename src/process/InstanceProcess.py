import time

from config import Config

from src.helpers.ReadPrpFile import ReadPrpFile as RD
from src.helpers.TargetsLoader import normalize_instance_file_key
from src.log.Logger import Logger
from src.process.ProcessResults import getResults
from src.helpers.SolverTelemetry import get_config, new_run_id, write_shards
from src.solvers.MultProductProdctionRoutingProblem import (
    MultProductProdctionRoutingProblem as MPPRP,
)
from src.solvers.ParticleSwarmOptimization import (
    ParticleSwarmOptimization as PSOSolver,
)
import gc


class InstanceProcess:

    def __init__(
        self,
        instance,
        output,
        numThreads=None,
        timeLimit=None,
        log: Logger = None,
        solver="GUROBY",
        weight=None,
        targets_by_file=None,
        alpha=None,
        task_context=None,
    ):
        self.instance = instance
        self.numThreads = numThreads
        self.timeLimit = timeLimit
        self.output = output
        self.isFinished = False
        self.solver = solver
        self.log: Logger = log
        self.weight = weight
        self.targets_by_file = targets_by_file
        self.alpha = alpha
        self.task_context = task_context or {}
        self.telemetry_config = get_config(Config)

    def solverInstancie(self, data):
        # Python 3.8-compatible replacement for match-case
        if self.solver == "GUROBY":
            self.log.debug("Solver: GUROBY")
            return MPPRP(
                map=data, dir=self.output, log=self.log, start={"start": False}
            )
        elif self.solver == "PSO":
            self.log.debug("Solver: PSO")
            return PSOSolver(map=data, dir=self.output, log=self.log)
        else:
            self.log.error(f"Solver não suportado: {self.solver}")
            return 0

    def process(self):

        data = None
        instance = None
        results = None
        Z = X = Y = I = R = Q = P = None
        FO = GAP = TIME = EPSILON = SOL_COUNT = RELAXED_MODEL_OBJE_VAL = None
        NODE_COUNT = OBJ_BOUND = NEW_TARGETS = None

        try:
            data = RD(file_path=self.instance, log=self.log).getDataSet()
            data["weight"] = self.weight
            data["alpha"] = self.alpha

            targets = []
            if self.targets_by_file:
                key = normalize_instance_file_key(data.get("file"))
                targets = self.targets_by_file.get(key, [])
            data["targets"] = targets

            strategy_started_at = time.time()
            instance = self.solverInstancie(data)

            solve_started_at = time.time()
            instance.solver(timeLimit=self.timeLimit, numThreads=self.numThreads)
            solve_elapsed_seconds = time.time() - solve_started_at

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
            if self.telemetry_config["enabled"] and hasattr(instance, "get_telemetry"):
                telemetry = instance.get_telemetry()
                strategy = telemetry.get("strategy", self.solver.lower())
                summary = {
                    "run_id": new_run_id(self.task_context, strategy),
                    "experiment_id": self.telemetry_config["experiment_id"],
                    "run_tag": self.task_context.get("run_tag"), "task_number": self.task_context.get("task_number"),
                    "instance_file": data.get("file"), "weight": str(data.get("weight")), "alpha": data.get("alpha"),
                    "threads": self.numThreads, "time_limit_seconds": self.timeLimit,
                    "solver_variant": strategy,
                    "pipeline_solver_seconds": solve_elapsed_seconds,
                    "total_seconds": time.time() - strategy_started_at,
                    **telemetry,
                }
                write_shards(self.output, summary, telemetry.get("pso_iterations") if self.telemetry_config["save_pso_iterations"] else [], telemetry.get("mip_events", []))
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
            )
            self.log.debug("Resultados gerados.")
            self.isFinished = True
        finally:
            if instance is not None and hasattr(instance, "terminate"):
                try:
                    instance.terminate()
                except Exception as e:
                    if self.log:
                        self.log.error(f"Erro ao liberar modelo: {e}")

            del results
            del Z, X, Y, I, R, Q, P
            del FO, GAP, TIME, EPSILON, SOL_COUNT, RELAXED_MODEL_OBJE_VAL
            del NODE_COUNT, OBJ_BOUND, NEW_TARGETS
            del instance
            del data
            gc.collect()
