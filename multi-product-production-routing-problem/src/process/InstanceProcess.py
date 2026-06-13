import pdb
import time

from src.helpers.GraphDisplay import graphResults
from src.helpers.JobGuardrails import JobGuardrails
from src.helpers.ReadPrpFile import ReadPrpFile as RD
from src.helpers.TargetsLoader import normalize_instance_file_key
from src.log.Logger import Logger
from src.process.ProcessResults import getResults, new_get_results
from src.solvers.MultProductProdctionRoutingProblem import (
    MultProductProdctionRoutingProblem as MPPRP,
)
from src.solvers.MultProductProdctionRoutingProblemGreedyConstructiveHeuristic import (
    MultProductProdctionRoutingProblemGreedyConstructiveHeuristic as MPPRPG,
)
import gc


class InstanceProcess:

    def __init__(
        self,
        instance,
        output,
        isPloat="false",
        numThreads=None,
        timeLimit=None,
        log: Logger = None,
        solver="GRASP",
        weight=None,
        targets_by_file=None,
        alpha=None,
        guardrail_runtime=None,
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

    def isProcessFinished(self):
        return self.isFinished

    def _log_phase(self, phase, started_at):
        if not self.guardrails.is_enabled() or not self.guardrails.log_solver_phase_timing:
            return

        snapshot = self.guardrails.snapshot()
        self.log.info(
            self.guardrails.format_snapshot(
                snapshot,
                context={
                    "event": f"phase_{phase}",
                    "instance_file": self.instance,
                },
            )
            + f" phase_elapsed={time.time() - started_at:.2f}s"
        )

    def solverInstancie(self, data):
        # Python 3.8-compatible replacement for match-case
        if self.solver == "GUROBY":
            self.log.info(f" Solver: GUROBY")
            return MPPRP(
                map=data, dir=self.output, log=self.log, start={"start": False}
            )
        elif self.solver == "HEURISTICA_CONSTRUTIVA_MIT_START":
            self.log.info(f" Solver: HEURISTICA_CONSTRUTIVA_MIT_START")
            instancia = MPPRPG(map=data, dir=self.output, log=self.log)
            instancia.setMitStart(True)
            return instancia
        elif self.solver == "HEURISTICA_CONSTRUTIVA":
            self.log.info(f" Solver: HEURISTICA_CONSTRUTIVA")
            return MPPRPG(map=data, dir=self.output, log=self.log)
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
            phase_started_at = time.time()
            data = RD(file_path=self.instance, log=self.log).getDataSet()
            self._log_phase("read_instance", phase_started_at)
            data["weight"] = self.weight
            data["alpha"] = self.alpha

            targets = []
            if self.targets_by_file:
                key = normalize_instance_file_key(data.get("file"))
                targets = self.targets_by_file.get(key, [])
            data["targets"] = targets

            phase_started_at = time.time()
            instance = self.solverInstancie(data)
            self._log_phase("build_solver", phase_started_at)

            phase_started_at = time.time()
            instance.solver(timeLimit=self.timeLimit, numThreads=self.numThreads)
            self._log_phase("solve", phase_started_at)

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
            self._log_phase("extract_results", phase_started_at)
            # FO, f1, f2, f3, f4, GAP, TIME, SOL_COUNT, RELAXED_MODEL_OBJE_VAL, NODE_COUNT, OBJ_BOUND = instance.new_get_results()

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
            )
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

            del results
            del Z, X, Y, I, R, Q, P
            del FO, GAP, TIME, EPSILON, SOL_COUNT, RELAXED_MODEL_OBJE_VAL
            del NODE_COUNT, OBJ_BOUND, NEW_TARGETS
            del instance
            del data
            gc.collect()
