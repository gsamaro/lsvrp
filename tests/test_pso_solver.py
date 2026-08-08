import sys
import types
import unittest
import inspect
from pathlib import Path
from unittest.mock import patch

import numpy as np


solver_module = types.ModuleType("src.solvers.MultProductProdctionRoutingProblem")
solver_module.MultProductProdctionRoutingProblem = object
sys.modules.setdefault("src.solvers.MultProductProdctionRoutingProblem", solver_module)

from src.process.InstanceProcess import InstanceProcess
from src.solvers.ParticleSwarmOptimization import ParticleSwarmOptimization
import src.solvers.ParticleSwarmOptimization as pso_module


class DummyLogger:
    def __init__(self):
        self.messages = []

    def info(self, message):
        self.messages.append(("info", message))

    def warning(self, message):
        self.messages.append(("warning", message))

    def error(self, message):
        self.messages.append(("error", message))


class PSOSolverTestCase(unittest.TestCase):
    def _minimal_data(self, demand=5, vehicle_capacity=10):
        return {
            "num_products": 1,
            "num_customers": 1,
            "num_periods": 1,
            "num_vehicles": 1,
            "B": 10,
            "b_p": [1],
            "c_p": [2],
            "s_p": [3],
            "M": 100,
            "U_pi": [[20, 20]],
            "I_pi0": [[0, 0]],
            "h_pi": [[0, 1]],
            "C": vehicle_capacity,
            "f": 7,
            "a_ik": [[0, 11], [11, 0]],
            "d_pit": [[[demand]]],
            "weight": [0.2] * 5,
            "alpha": 0.01,
        }

    def test_pso_runs_iterations_and_tracks_best_solution(self):
        values = {
            ("solver", "pso"): {
                "swarm_size": 4,
                "max_iterations": 2,
                "seed": 123,
                "use_as_mip_start": False,
                "return_heuristic_result_without_cplex": True,
            }
        }

        with patch("src.solvers.ParticleSwarmOptimization.Config.get_nested") as get_nested:
            get_nested.side_effect = lambda *keys, default=None: values.get(keys, default)
            solver = ParticleSwarmOptimization(
                map=self._minimal_data(),
                dir="/tmp",
                log=DummyLogger(),
            )
            solver.solver()

        self.assertIsNotNone(solver.global_best_solution)
        self.assertEqual(solver.positions.shape[0], 4)
        self.assertTrue(np.isfinite(solver.global_best_cost))
        np.testing.assert_allclose(
            solver.heuristic.lower_bounds,
            solver.relaxed_total_bounds["lower_solution"]["X"],
        )
        np.testing.assert_allclose(
            solver.heuristic.upper_bounds,
            solver.relaxed_total_bounds["upper_solution"]["X"],
        )
        np.testing.assert_allclose(
            solver.heuristic.lower_delivery_bounds,
            solver.relaxed_total_bounds["lower_solution"]["Q"],
        )
        np.testing.assert_allclose(
            solver.heuristic.upper_delivery_bounds,
            solver.relaxed_total_bounds["upper_solution"]["Q"],
        )
        self.assertIsNotNone(solver.relaxed_base)
        np.testing.assert_allclose(
            solver.relaxed_base["X"],
            solver.relaxed_total_bounds["lower_solution"]["X"],
        )
        log_messages = [message for _, message in solver.log.messages]
        self.assertTrue(
            any(
                "PSO initialization" in message
                and "feasible=4/4" in message
                and "x_profiles=" in message
                and "q_profiles=" in message
                for message in log_messages
            )
        )
        self.assertTrue(
            any(
                "PSO iteration=1" in message
                and "x_profiles=" in message
                and "q_profiles=" in message
                for message in log_messages
            )
        )
        self.assertTrue(
            any(
                "PSO final " in message
                and "best_first_seen_iteration=" in message
                and "final_feasible=4/4" in message
                for message in log_messages
            )
        )
        self.assertTrue(
            any(
                "PSO final diagnostics " in message
                and "pre_repair_infeasible=" in message
                and "resample_infeasible=" in message
                and "x_profiles=" in message
                and "q_profiles=" in message
                for message in log_messages
            )
        )

    def test_pso_passes_warm_start_to_exact_solver(self):
        starts = []

        class FakeExactSolver:
            def __init__(self, map, dir, log, start):
                starts.append(start)

            def solver(self, timeLimit=None, numThreads=None):
                pass

            def getResults(self):
                return ("exact",)

            def terminate(self):
                pass

        values = {
            ("solver", "pso"): {
                "swarm_size": 3,
                "max_iterations": 1,
                "seed": 123,
                "use_as_mip_start": True,
                "return_heuristic_result_without_cplex": False,
            }
        }

        with patch.object(pso_module, "MPPRP", FakeExactSolver):
            with patch("src.solvers.ParticleSwarmOptimization.Config.get_nested") as get_nested:
                get_nested.side_effect = lambda *keys, default=None: values.get(keys, default)
                solver = ParticleSwarmOptimization(
                    map=self._minimal_data(),
                    dir="/tmp",
                    log=DummyLogger(),
                )
                solver.solver()

        self.assertEqual(len(starts), 1)
        self.assertTrue(starts[0]["start"])
        self.assertIn("variables", starts[0])

    def test_pso_returns_standalone_results(self):
        values = {
            ("solver", "pso"): {
                "swarm_size": 2,
                "max_iterations": 1,
                "seed": 123,
                "use_as_mip_start": False,
                "return_heuristic_result_without_cplex": True,
            }
        }

        with patch("src.solvers.ParticleSwarmOptimization.Config.get_nested") as get_nested:
            get_nested.side_effect = lambda *keys, default=None: values.get(keys, default)
            solver = ParticleSwarmOptimization(
                map=self._minimal_data(),
                dir="/tmp",
                log=DummyLogger(),
            )
            solver.solver()
            results = solver.getResults()

        self.assertEqual(len(results), 16)
        self.assertGreater(results[7], 0)

    def test_pso_python_reference_backend_remains_available(self):
        values = {
            ("solver", "pso"): {
                "swarm_size": 3,
                "max_iterations": 1,
                "seed": 123,
                "execution_backend": "python",
                "parallel_workers": 1,
                "use_as_mip_start": False,
                "return_heuristic_result_without_cplex": True,
            }
        }
        with patch("src.solvers.ParticleSwarmOptimization.Config.get_nested") as get_nested:
            get_nested.side_effect = lambda *keys, default=None: values.get(keys, default)
            solver = ParticleSwarmOptimization(
                map=self._minimal_data(), dir="/tmp", log=DummyLogger()
            )
            solver.solver()

        self.assertEqual(solver.pso_config["execution_backend"], "python")
        self.assertEqual(int(solver.population_state.feasible.sum()), 3)

    def test_adaptive_coefficients_reach_configured_extremes_and_velocity_is_limited(self):
        values = {
            ("solver", "pso"): {
                "swarm_size": 4,
                "max_iterations": 3,
                "seed": 123,
                "mutation_particle_rate": 0,
                "stagnation_patience": 99,
                "return_heuristic_result_without_cplex": True,
            }
        }
        with patch("src.solvers.ParticleSwarmOptimization.Config.get_nested") as get_nested:
            get_nested.side_effect = lambda *keys, default=None: values.get(keys, default)
            solver = ParticleSwarmOptimization(
                map=self._minimal_data(), dir="/tmp", log=DummyLogger()
            )

        self.assertEqual(solver._schedule_value("inertia_initial", "inertia_final", 1), 0.9)
        self.assertEqual(solver._schedule_value("inertia_initial", "inertia_final", 3), 0.4)
        solver.positions = np.zeros((4, solver.heuristic.particle_dim))
        solver.velocities = np.full_like(solver.positions, 100.0)
        solver.personal_best_positions = np.zeros_like(solver.positions)
        solver.personal_best_costs = np.arange(4, dtype=float)
        solver.global_best_position = np.zeros(solver.heuristic.particle_dim)
        solver.personal_best_solutions = [None] * 4
        solver.solutions = [{"feasible": True}] * 4

        solver._update_positions(iteration=1)

        self.assertLessEqual(np.max(np.abs(solver.velocities)), 2.0)

    def test_mutation_preserves_elite_and_reinitialization_replaces_worst(self):
        values = {
            ("solver", "pso"): {
                "swarm_size": 10,
                "max_iterations": 10,
                "seed": 123,
                "mutation_particle_rate": 0.5,
                "elite_fraction": 0.1,
                "reinitialize_fraction": 0.2,
                "stagnation_patience": 5,
                "return_heuristic_result_without_cplex": True,
            }
        }
        with patch("src.solvers.ParticleSwarmOptimization.Config.get_nested") as get_nested:
            get_nested.side_effect = lambda *keys, default=None: values.get(keys, default)
            solver = ParticleSwarmOptimization(
                map=self._minimal_data(), dir="/tmp", log=DummyLogger()
            )

        solver.positions = np.zeros((10, solver.heuristic.particle_dim))
        solver.velocities = np.zeros_like(solver.positions)
        solver.personal_best_positions = np.zeros_like(solver.positions)
        solver.personal_best_costs = np.arange(10, dtype=float)
        solver.personal_best_solutions = [{"feasible": True}] * 10
        solver.solutions = [{"feasible": True}] * 10

        solver._mutate_non_elites(iteration=1)

        self.assertEqual(solver._mutated_particles, 5)
        self.assertTrue(np.all(solver.positions[0] == 0))
        self.assertGreater(np.count_nonzero(solver.positions), 0)

        solver._stagnation_count = 5
        solver._reinitialize_if_stagnant()

        self.assertEqual(solver._reinitialized_particles, 2)
        self.assertTrue(np.all(solver.positions[0] == 0))
        self.assertTrue(np.isinf(solver.personal_best_costs[8]))
        self.assertTrue(np.isinf(solver.personal_best_costs[9]))
        self.assertFalse(solver.solutions[8]["feasible"])
        self.assertFalse(solver.solutions[9]["feasible"])

    def test_parallel_workers_auto_uses_eight_locally_and_one_under_mpi(self):
        values = {
            ("solver", "pso"): {
                "swarm_size": 2,
                "max_iterations": 1,
                "execution_backend": "python",
                "parallel_workers": "auto",
                "lot_sizing_bounds": {"enabled": False},
                "return_heuristic_result_without_cplex": True,
            }
        }
        with patch("src.solvers.ParticleSwarmOptimization.Config.get_nested") as get_nested:
            get_nested.side_effect = lambda *keys, default=None: values.get(keys, default)
            with patch.dict("os.environ", {}, clear=True), patch(
                "src.solvers.ParticleSwarmOptimization.os.cpu_count", return_value=10
            ):
                local = ParticleSwarmOptimization(
                    map=self._minimal_data(), dir="/tmp", log=DummyLogger()
                )
            with patch.dict("os.environ", {"OMPI_COMM_WORLD_SIZE": "16"}, clear=True):
                mpi = ParticleSwarmOptimization(
                    map=self._minimal_data(), dir="/tmp", log=DummyLogger()
                )

        self.assertEqual(local.parallel_workers, 8)
        self.assertEqual(mpi.parallel_workers, 1)

    def test_instance_process_selects_only_pso_solver(self):
        fake_instance = object()
        process = InstanceProcess(
            instance="./data/test.dat",
            output="./out/test/",
            log=DummyLogger(),
            solver="PSO",
        )

        with patch("src.process.InstanceProcess.PSOSolver", return_value=fake_instance) as factory:
            instance = process.solverInstancie(self._minimal_data())

        self.assertIs(instance, fake_instance)
        factory.assert_called_once()

    def test_legacy_solver_files_are_removed(self):
        project_root = Path(__file__).resolve().parents[1]
        removed = [
            "src/solvers/MultProductProdctionRoutingProblemGreedyConstructiveHeuristic.py",
            "src/solvers/MultProductProdctionRoutingProblemParticleSwarmOptimization.py",
            "src/solvers/MultProductProductionRoutingProblemGrasp.py",
            "src/solvers/PSO.py",
            "src/solvers/SolutionOperators.py",
            "src/solvers/GreedyRandomizedConstructionRoute.py",
            "src/solvers/TwoOptOnRoute.py",
        ]

        for relative_path in removed:
            self.assertFalse((project_root / relative_path).exists(), relative_path)

    def test_instance_process_and_pso_do_not_reference_removed_modules(self):
        import src.process.InstanceProcess as instance_process_module

        instance_source = inspect.getsource(instance_process_module)
        pso_source = inspect.getsource(pso_module)

        forbidden = [
            "MultProductProdctionRoutingProblemGreedyConstructiveHeuristic",
            "MultProductProdctionRoutingProblemParticleSwarmOptimization",
            "SolutionOperators",
            "GreedyRandomizedConstructionRoute",
            "TwoOptOnRoute",
            "MultProductProductionRoutingProblemGrasp",
            '"HEURISTICA_CONSTRUTIVA"',
        ]

        for token in forbidden:
            self.assertNotIn(token, instance_source)
            self.assertNotIn(token, pso_source)


if __name__ == "__main__":
    unittest.main()
