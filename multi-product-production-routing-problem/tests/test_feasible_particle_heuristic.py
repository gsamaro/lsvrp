import unittest

import numpy as np

from src.solvers.FeasibleParticleHeuristic import FeasibleParticleHeuristic


class DummyLogger:
    def info(self, message):
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


if __name__ == "__main__":
    unittest.main()
