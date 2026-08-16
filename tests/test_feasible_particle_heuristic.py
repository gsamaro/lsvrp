import unittest

import numpy as np

from src.solvers.FeasibleParticleHeuristic import FeasibleParticleHeuristic
from src.solvers._solver_common import evaluate_solution_cost


class DummyLogger:
    def info(self, message):
        pass

    def debug(self, message):
        pass

    def warning(self, message):
        pass

    def error(self, message):
        pass


class FeasibleParticleHeuristicTestCase(unittest.TestCase):
    def _minimal_data(self):
        return {
            "num_products": 1,
            "num_customers": 1,
            "num_periods": 2,
            "num_vehicles": 1,
            "B": 10,
            "b_p": [1],
            "c_p": [2],
            "s_p": [3],
            "M": 100,
            "U_pi": [[20, 20]],
            "I_pi0": [[0, 0]],
            "h_pi": [[0, 1]],
            "C": 10,
            "f": 7,
            "a_ik": [[0, 11], [11, 0]],
            "d_pit": [[[2, 3]]],
            "weight": [0.2] * 5,
            "alpha": 0.01,
        }

    def test_build_population_creates_feasible_solution(self):
        heuristic = FeasibleParticleHeuristic(
            map=self._minimal_data(),
            dir="/tmp",
            log=DummyLogger(),
        )
        particles = np.full((2, heuristic.particle_dim), -100.0, dtype=float)

        solutions = heuristic.build_population(particles)

        self.assertEqual(len(solutions), 2)
        self.assertTrue(all(solution["feasible"] for solution in solutions))
        self.assertEqual(solutions[0]["X"][0, 0], 2)
        self.assertEqual(solutions[0]["Q"][0, 0, 1, 0], 2)

    def test_last_period_forces_remaining_demand_delivery(self):
        heuristic = FeasibleParticleHeuristic(
            map=self._minimal_data(),
            dir="/tmp",
            log=DummyLogger(),
        )
        particle = np.full((1, heuristic.particle_dim), -100.0, dtype=float)

        solution = heuristic.build_population(particle)[0]

        self.assertEqual(solution["Q"][0, 0, 1, 1], 3)
        self.assertEqual(solution["I"][0, 1, 1], 0)

    def test_vehicle_infeasible_instance_returns_infeasible_solution(self):
        data = self._minimal_data()
        data["C"] = 2
        data["d_pit"] = [[[5, 5]]]
        heuristic = FeasibleParticleHeuristic(map=data, dir="/tmp", log=DummyLogger())
        particle = np.zeros((1, heuristic.particle_dim), dtype=float)

        solution = heuristic.build_population(particle)[0]

        self.assertFalse(solution["feasible"])

    def test_repair_population_rebuilds_feasible_solution(self):
        heuristic = FeasibleParticleHeuristic(
            map=self._minimal_data(),
            dir="/tmp",
            log=DummyLogger(),
        )
        particles = np.ones((1, heuristic.particle_dim), dtype=float) * -100

        repaired = heuristic.repair_population(particles)[0]
        report = heuristic.validate_solution(repaired)

        self.assertTrue(repaired["feasible"])
        self.assertTrue(report["feasible"], report["violations"])

    def test_repair_population_reuses_previous_feasible_solution(self):
        heuristic = FeasibleParticleHeuristic(
            map=self._minimal_data(),
            dir="/tmp",
            log=DummyLogger(),
        )
        previous = heuristic.build_population(
            np.zeros((1, heuristic.particle_dim), dtype=float)
        )
        heuristic._build_solution_from_particle = lambda row: {"feasible": False}
        repaired = heuristic.repair_population(
            np.zeros((1, heuristic.particle_dim), dtype=float),
            previous_solutions=previous,
        )[0]

        self.assertTrue(repaired["feasible"])
        self.assertTrue(repaired["reused_previous"])

    def test_bounds_are_drained_to_customers_with_diverse_delivery_profiles(self):
        data = {
            "num_products": 1,
            "num_customers": 2,
            "num_periods": 1,
            "num_vehicles": 1,
            "B": 10,
            "b_p": [1],
            "c_p": [2],
            "s_p": [3],
            "M": 100,
            "U_pi": [[0, 4, 4]],
            "I_pi0": [[0, 0, 0]],
            "h_pi": [[0, 1, 1]],
            "C": 10,
            "f": 7,
            "a_ik": [[0, 1, 2], [1, 0, 1], [2, 1, 0]],
            "d_pit": [[[1], [1]]],
            "weight": [0.2] * 5,
            "alpha": 0.01,
        }
        q_lower = np.zeros((1, 1, 3, 1))
        q_upper = np.zeros((1, 1, 3, 1))
        q_upper[0, 0, 1, 0] = 5
        q_upper[0, 0, 2, 0] = 5
        bounds = {
            "lower": np.array([[8.0]]),
            "upper": np.array([[8.0]]),
            "q_lower": q_lower,
            "q_upper": q_upper,
        }
        heuristic = FeasibleParticleHeuristic(data, "/tmp", DummyLogger(), bounds=bounds)

        particles = np.array(
            [
                [0.0, 0.0, 10.0, -10.0],
                [0.0, 0.0, -10.0, 10.0],
            ]
        )
        solutions = heuristic.build_population(particles)

        self.assertTrue(all(solution["feasible"] for solution in solutions))
        self.assertTrue(all(solution["I"][0, 0, 0] == 0 for solution in solutions))
        self.assertTrue(
            all(
                solution["I"][0, customer, 0] <= data["U_pi"][0][customer]
                for solution in solutions
                for customer in (1, 2)
            )
        )
        self.assertTrue(
            all(
                sum(solution["Q"][0, 0, customer, 0] for customer in (1, 2))
                == 8
                for solution in solutions
            )
        )
        self.assertNotEqual(
            solutions[0]["Q"][0, 0, 1, 0],
            solutions[1]["Q"][0, 0, 1, 0],
        )

    def test_vehicle_aware_drain_keeps_each_customer_in_one_vehicle(self):
        data = {
            "num_products": 1,
            "num_customers": 5,
            "num_periods": 1,
            "num_vehicles": 2,
            "B": 1000,
            "b_p": [1],
            "c_p": [2],
            "s_p": [3],
            "M": 1000,
            "U_pi": [[0, 300, 300, 300, 300, 300]],
            "I_pi0": [[0, 0, 0, 0, 0, 0]],
            "h_pi": [[0, 1, 1, 1, 1, 1]],
            "C": 300,
            "f": 7,
            "a_ik": [[0 for _ in range(6)] for _ in range(6)],
            "d_pit": [[[100], [100], [100], [100], [100]]],
            "weight": [0.2] * 5,
            "alpha": 0.01,
        }
        bounds = {
            "lower": np.array([[583.0]]),
            "upper": np.array([[583.0]]),
        }
        heuristic = FeasibleParticleHeuristic(data, "/tmp", DummyLogger(), bounds=bounds)

        solution = heuristic.build_population(
            np.zeros((1, heuristic.particle_dim), dtype=float)
        )[0]

        self.assertTrue(solution["feasible"])
        self.assertEqual(solution["I"][0, 0, 0], 0)
        for vehicle in range(data["num_vehicles"]):
            self.assertLessEqual(
                sum(solution["Q"][0, vehicle, customer, 0] for customer in range(1, 6)),
                data["C"],
            )
        for customer in range(1, 6):
            self.assertLessEqual(
                sum(solution["Q"][0, vehicle, customer, 0] > 0 for vehicle in range(2)),
                1,
            )

    def test_normalize_gene_is_stable_for_large_values(self):
        heuristic = FeasibleParticleHeuristic(
            map=self._minimal_data(),
            dir="/tmp",
            log=DummyLogger(),
        )

        self.assertGreater(heuristic._normalize_gene(1_000), 0.999)
        self.assertLess(heuristic._normalize_gene(-1_000), 0.001)

    def test_compact_solution_materializes_routes_and_preserves_cost(self):
        heuristic = FeasibleParticleHeuristic(
            map=self._minimal_data(), dir="/tmp", log=DummyLogger()
        )
        compact = heuristic.build_population(
            np.zeros((1, heuristic.particle_dim), dtype=float)
        )[0]

        self.assertNotIn("R", compact)
        self.assertNotIn("Z", compact)
        self.assertTrue(heuristic.validate_compact_solution(compact)["feasible"])

        full = heuristic.materialize_solution(compact)
        self.assertIn("R", full)
        self.assertIn("Z", full)
        self.assertTrue(heuristic.validate_solution(full)["feasible"])
        self.assertEqual(compact["cost"], evaluate_solution_cost(heuristic.problem, full))

    def test_fast_and_complete_validation_reject_inventory_violation(self):
        heuristic = FeasibleParticleHeuristic(
            map=self._minimal_data(), dir="/tmp", log=DummyLogger()
        )
        compact = heuristic.build_population(
            np.zeros((1, heuristic.particle_dim), dtype=float)
        )[0]
        compact["I"][0, 0, 0] = 999

        fast = heuristic.validate_compact_solution(compact)
        complete = heuristic.validate_solution(compact)

        self.assertFalse(fast["feasible"])
        self.assertFalse(complete["feasible"])

    def test_fast_validation_rejects_route_inconsistent_with_delivery(self):
        heuristic = FeasibleParticleHeuristic(
            map=self._minimal_data(), dir="/tmp", log=DummyLogger()
        )
        compact = heuristic.build_population(
            np.zeros((1, heuristic.particle_dim), dtype=float)
        )[0]
        compact["route_plan"][0][0] = []

        report = heuristic.validate_compact_solution(compact)

        self.assertFalse(report["feasible"])
        self.assertTrue(
            any("rota inconsistente" in violation for violation in report["violations"])
        )

    def test_numba_backend_matches_python_backend(self):
        data = self._minimal_data()
        python = FeasibleParticleHeuristic(
            data, "/tmp", DummyLogger(), execution_backend="python"
        )
        numba = FeasibleParticleHeuristic(
            data,
            "/tmp",
            DummyLogger(),
            execution_backend="numba",
            parallel_workers=1,
        )
        positions = np.random.default_rng(77).normal(
            size=(8, python.particle_dim)
        )

        expected = python.build_population_state(positions)
        actual = numba.build_population_state(positions)

        for name in (
            "X",
            "Y",
            "I",
            "Q",
            "assignments",
            "route_nodes",
            "route_lengths",
            "feasible",
            "costs",
        ):
            np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))

    def test_route_distance_tie_uses_q_gene_then_customer_index(self):
        data = {
            "num_products": 1,
            "num_customers": 2,
            "num_periods": 1,
            "num_vehicles": 1,
            "B": 10,
            "b_p": [1],
            "c_p": [1],
            "s_p": [1],
            "M": 100,
            "U_pi": [[10, 10, 10]],
            "I_pi0": [[0, 0, 0]],
            "h_pi": [[0, 0, 0]],
            "C": 10,
            "f": 1,
            "a_ik": [[0, 5, 5], [5, 0, 1], [5, 1, 0]],
            "d_pit": [[[1], [1]]],
            "weight": [0.2] * 5,
            "alpha": 0.01,
        }
        for backend in ("python", "numba"):
            heuristic = FeasibleParticleHeuristic(
                data,
                "/tmp",
                DummyLogger(),
                execution_backend=backend,
                parallel_workers=1,
            )
            position = np.array([[0.0, 0.0, -2.0, 2.0]])
            solution = heuristic.build_population(position)[0]
            self.assertEqual(solution["route_plan"][0][0], [2, 1])

            tied_position = np.zeros((1, heuristic.particle_dim))
            tied_solution = heuristic.build_population(tied_position)[0]
            self.assertEqual(tied_solution["route_plan"][0][0], [1, 2])

    def test_numba_backend_is_deterministic_across_thread_counts(self):
        data = self._minimal_data()
        one_thread = FeasibleParticleHeuristic(
            data,
            "/tmp",
            DummyLogger(),
            execution_backend="numba",
            parallel_workers=1,
        )
        positions = np.random.default_rng(91).normal(
            size=(16, one_thread.particle_dim)
        )
        expected = one_thread.build_population_state(positions)
        eight_threads = FeasibleParticleHeuristic(
            data,
            "/tmp",
            DummyLogger(),
            execution_backend="numba",
            parallel_workers=8,
        )
        actual = eight_threads.build_population_state(positions)

        for name in ("X", "I", "Q", "route_nodes", "route_lengths", "costs"):
            np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))


if __name__ == "__main__":
    unittest.main()
