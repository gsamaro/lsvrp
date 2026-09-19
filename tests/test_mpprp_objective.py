import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.solvers.RoundedCapacitySeparation import RoundedCapacityCut


def _load_real_mpprp():
    module_name = "mpprp_real_objective_test"
    if module_name in sys.modules:
        return sys.modules[module_name]

    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    module_path = project_root / "src/solvers/MultProductProdctionRoutingProblem.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    sys.modules[module_name] = module
    return module


class DummyLogger:
    def info(self, message):
        pass

    def debug(self, message):
        pass

    def warning(self, message):
        pass

    def error(self, message):
        pass


class MPPRPObjectiveTestCase(unittest.TestCase):
    @staticmethod
    def _data(num_customers=1, num_vehicles=1):
        nodes = num_customers + 1
        return {
            "num_products": 1,
            "num_customers": num_customers,
            "num_periods": 1,
            "num_vehicles": num_vehicles,
            "B": 10,
            "b_p": [1],
            "c_p": [2],
            "s_p": [3],
            "M": 100,
            "U_pi": [[20] * nodes],
            "I_pi0": [[0] * nodes],
            "h_pi": [[0] + [1] * num_customers],
            "C": 10,
            "f": 7,
            "a_ik": [[0 if i == k else 11 for k in range(nodes)] for i in range(nodes)],
            "d_pit": [[[5] for _ in range(num_customers)]],
            "weight": [0.2] * 5,
            "alpha": 0.01,
        }

    def test_singleobjective_objective_builds_without_list_expression_error(self):
        data = self._data()

        module = _load_real_mpprp()
        solver = module.MultProductProdctionRoutingProblem(
            map=data,
            dir="/tmp",
            log=DummyLogger(),
            start={"start": False},
        )
        solver.createDecisionVariables()

        with patch("mpprp_real_objective_test.Config.get_nested") as get_nested:
            values = {
                ("postprocessing", "build_target"): False,
                ("solver", "multiobjective"): False,
            }
            get_nested.side_effect = lambda *keys, default=None: values.get(keys, default)
            solver.crateObjectiveFunction()

        self.assertIsNotNone(solver.model.objective_expr)

    def test_multiobjective_counts_lambda_once_outside_period_component_sum(self):
        data = self._data()
        data["targets"] = [{f"f{j}_target": 10.0 for j in range(1, 6)}]

        module = _load_real_mpprp()
        solver = module.MultProductProdctionRoutingProblem(
            map=data,
            dir="/tmp",
            log=DummyLogger(),
            start={"start": False},
        )
        solver.createDecisionVariables()

        with patch("mpprp_real_objective_test.Config.get_nested") as get_nested:
            values = {
                ("postprocessing", "build_target"): False,
                ("solver", "multiobjective"): True,
            }
            get_nested.side_effect = lambda *keys, default=None: values.get(keys, default)
            solver.crateObjectiveFunction()

        self.assertAlmostEqual(
            solver.model.objective_expr.get_coef(solver.lambda_),
            data["alpha"],
        )

    def test_positive_only_deviations_omit_negative_variables_and_keep_goal_upper_bound(self):
        data = self._data()
        data["targets"] = [{f"f{j}_target": 10.0 for j in range(1, 6)}]

        module = _load_real_mpprp()
        solver = module.MultProductProdctionRoutingProblem(
            map=data,
            dir="/tmp",
            log=DummyLogger(),
            start={"start": False},
            positive_only_deviations=True,
        )
        solver.createDecisionVariables()
        solver.crateObjectiveFunction()
        solver.createGoalProgrammingRestrictions()

        assert solver.negative is None
        names = [variable.name for variable in solver.model.iter_variables()]
        assert not any(name.startswith("n_") for name in names)
        constraints = list(solver.model.iter_constraints())
        assert len(constraints) == 5
        assert all(">=" in str(constraint) for constraint in constraints)

    def test_default_goal_programming_omits_negative_variables(self):
        data = self._data()
        data["targets"] = [{f"f{j}_target": 10.0 for j in range(1, 6)}]

        module = _load_real_mpprp()
        solver = module.MultProductProdctionRoutingProblem(
            map=data,
            dir="/tmp",
            log=DummyLogger(),
            start={"start": False},
        )
        solver.createDecisionVariables()

        assert solver.positive_only_deviations is True
        assert solver.negative is None

    def test_decision_variables_exclude_diagonal_arcs(self):
        module = _load_real_mpprp()
        solver = module.MultProductProdctionRoutingProblem(
            map=self._data(num_customers=2),
            dir="/tmp",
            log=DummyLogger(),
            start={"start": False},
        )

        solver.createDecisionVariables()

        self.assertEqual(6, len(solver.model.Z_v_i_k_t))
        self.assertEqual(6, len(solver.model.R_p_v_i_k_t))
        for node in range(3):
            self.assertNotIn((0, node, node, 0), solver.model.Z_v_i_k_t)
            self.assertNotIn((0, 0, node, node, 0), solver.model.R_p_v_i_k_t)

    def test_strengthened_bounds_replace_global_production_big_m(self):
        module = _load_real_mpprp()
        data = self._data(num_customers=2, num_vehicles=2)
        data["M"] = 100
        data["B"] = 10
        data["d_pit"] = [[[5], [7]]]
        solver = module.MultProductProdctionRoutingProblem(
            map=data,
            dir="/tmp",
            log=DummyLogger(),
            start={"start": False},
        )

        solver.createDecisionVariables()
        solver.createRelationshipBetweenProduction()

        names = [constraint.name for constraint in solver.model.iter_constraints()]
        assert "EQ_5_p_0_t_0" in names
        production_constraint = next(
            constraint for constraint in solver.model.iter_constraints()
            if constraint.name == "EQ_5_p_0_t_0"
        )
        assert "10" in str(production_constraint)

    def test_disabled_strengthened_bounds_use_global_production_big_m(self):
        module = _load_real_mpprp()
        data = self._data(num_customers=2, num_vehicles=2)
        data["strengthened_bounds"] = False
        data["M"] = 100
        data["B"] = 10
        data["d_pit"] = [[[5], [7]]]
        solver = module.MultProductProdctionRoutingProblem(
            map=data,
            dir="/tmp",
            log=DummyLogger(),
            start={"start": False},
        )

        solver.createDecisionVariables()
        solver.createRelationshipBetweenProduction()

        production_constraint = next(
            constraint for constraint in solver.model.iter_constraints()
            if constraint.name == "EQ_5_p_0_t_0"
        )
        assert "100" in str(production_constraint)

    def test_disabled_strengthened_bounds_use_original_inventory_capacity(self):
        module = _load_real_mpprp()
        data = self._data(num_customers=1, num_vehicles=3)
        data["strengthened_bounds"] = False
        data["U_pi"] = [[100, 20]]
        data["I_pi0"] = [[7, 2]]
        data["d_pit"] = [[[1]]]
        solver = module.MultProductProdctionRoutingProblem(
            map=data,
            dir="/tmp",
            log=DummyLogger(),
            start={"start": False},
        )

        solver.createDecisionVariables()
        solver.createDelimitMaximumCapacityItemsAtPlant()

        constraints = {
            constraint.name: str(constraint)
            for constraint in solver.model.iter_constraints()
        }
        assert "I_0_1_0 <= 20.0" in constraints["EQ_6_p_0_i_1_t_0"]

    def test_inventory_constraints_use_tightened_customer_bounds(self):
        module = _load_real_mpprp()
        data = self._data(num_customers=1, num_vehicles=3)
        data["num_periods"] = 2
        data["C"] = 5
        data["U_pi"] = [[100, 20]]
        data["I_pi0"] = [[7, 2]]
        data["d_pit"] = [[[1, 3]]]
        solver = module.MultProductProdctionRoutingProblem(
            map=data,
            dir="/tmp",
            log=DummyLogger(),
            start={"start": False},
        )

        solver.createDecisionVariables()
        solver.createDelimitMaximumCapacityItemsAtPlant()

        constraints = {
            constraint.name: str(constraint)
            for constraint in solver.model.iter_constraints()
        }
        assert "I_0_1_0 <= 6.0" in constraints["EQ_6_p_0_i_1_t_0"]
        assert "I_0_1_1 <= 8.0" in constraints["EQ_6_p_0_i_1_t_1"]
        # The plant uses its original production-reachability bound.
        assert "I_0_0_0 <= 11.0" in constraints["EQ_6_p_0_i_0_t_0"]

    def test_z_delivery_bounds_are_always_present_without_w_variables(self):
        module = _load_real_mpprp()
        data = self._data(num_customers=2, num_vehicles=2)
        data["U_pi"] = [[3, 4, 5]]
        data["d_pit"] = [[[5], [7]]]
        solver = module.MultProductProdctionRoutingProblem(
            map=data,
            dir="/tmp",
            log=DummyLogger(),
            start={"start": False},
        )

        solver.createDecisionVariables()
        names = [variable.name for variable in solver.model.iter_variables()]
        assert not any(name.startswith("W_") for name in names)
        assert solver.createVehicleVisitDeliveryBounds() is True

        names = [constraint.name for constraint in solver.model.iter_constraints()]
        assert sum(name.startswith("VISIT_CAP_Z_") for name in names) == 4
        assert sum(name.startswith("VISIT_Q_Z_") for name in names) == 4

        q_constraint = next(
            constraint
            for constraint in solver.model.iter_constraints()
            if constraint.name == "VISIT_Q_Z_p_0_v_0_i_1_t_0"
        )
        assert "9" in str(q_constraint)

    def test_mip_start_ignores_diagonal_arc_values(self):
        module = _load_real_mpprp()
        nodes = 3
        solver = module.MultProductProdctionRoutingProblem(
            map=self._data(num_customers=2),
            dir="/tmp",
            log=DummyLogger(),
            start={
                "start": True,
                "variables": {
                    "X": [[0]],
                    "Y": [[0]],
                    "I": [[[0] for _ in range(nodes)]],
                    "Q": [[[[0] for _ in range(nodes)]]],
                    "R": [[[[[0] for _ in range(nodes)] for _ in range(nodes)]]],
                    "Z": [[[[0] for _ in range(nodes)] for _ in range(nodes)]],
                },
            },
        )
        solver.createDecisionVariables()

        solver.startVariables()

        self.assertEqual(1, solver.model.number_of_mip_starts)

    def test_solution_extraction_has_no_dead_route_diagnostics(self):
        module_path = Path(__file__).resolve().parents[1] / "src/solvers/MultProductProdctionRoutingProblem.py"
        source = module_path.read_text(encoding="utf-8")

        self.assertNotIn('string += str(Z[t][v][i][k])', source)
        self.assertNotIn('self.log.info("============ Z ================")', source)

    def test_symmetry_breaking_is_disabled_by_default(self):
        module = _load_real_mpprp()
        solver = module.MultProductProdctionRoutingProblem(
            map=self._data(num_customers=2, num_vehicles=3),
            dir="/tmp",
            log=DummyLogger(),
            start={"start": False},
        )
        solver.createDecisionVariables()

        with patch("mpprp_real_objective_test.Config.get_nested") as get_nested:
            get_nested.return_value = False
            solver.createVehicleSymmetryBreaking()

        assert not list(solver.model.iter_constraints())

    def test_symmetry_breaking_adds_vc_and_hc1_constraints(self):
        module = _load_real_mpprp()
        solver = module.MultProductProdctionRoutingProblem(
            map=self._data(num_customers=2, num_vehicles=3),
            dir="/tmp",
            log=DummyLogger(),
            start={"start": False},
        )
        solver.createDecisionVariables()
        with patch("mpprp_real_objective_test.Config.get_nested", return_value=True):
            solver.createVehicleSymmetryBreaking()

        names = [constraint.name for constraint in solver.model.iter_constraints()]

        assert sum(name.startswith("SB_VC_") for name in names) == 2
        assert sum(name.startswith("SB_HC1_") for name in names) == 4

    def test_symmetry_breaking_boundary_sizes(self):
        module = _load_real_mpprp()
        for num_vehicles, num_customers, expected_vc, expected_hc1 in [
            (1, 2, 0, 0),
            (3, 1, 2, 2),
        ]:
            solver = module.MultProductProdctionRoutingProblem(
                map=self._data(
                    num_customers=num_customers, num_vehicles=num_vehicles
                ),
                dir="/tmp",
                log=DummyLogger(),
                start={"start": False},
            )
            solver.createDecisionVariables()
            with patch("mpprp_real_objective_test.Config.get_nested", return_value=True):
                solver.createVehicleSymmetryBreaking()

            names = [constraint.name for constraint in solver.model.iter_constraints()]

            assert sum(name.startswith("SB_VC_") for name in names) == expected_vc
            assert sum(name.startswith("SB_HC1_") for name in names) == expected_hc1

    def test_symmetry_breaking_override_disables_configured_constraints(self):
        module = _load_real_mpprp()
        solver = module.MultProductProdctionRoutingProblem(
            map=self._data(num_customers=2, num_vehicles=3),
            dir="/tmp",
            log=DummyLogger(),
            start={"start": False},
            symmetry_breaking_hc1=False,
        )
        solver.createDecisionVariables()

        with patch("mpprp_real_objective_test.Config.get_nested", return_value=True):
            solver.createVehicleSymmetryBreaking()

        assert not list(solver.model.iter_constraints())

    def test_coelho_inequalities_are_disabled_by_default(self):
        module = _load_real_mpprp()
        solver = module.MultProductProdctionRoutingProblem(
            map=self._data(num_customers=2, num_vehicles=3),
            dir="/tmp",
            log=DummyLogger(),
            start={"start": False},
        )
        solver.createDecisionVariables()

        with patch("mpprp_real_objective_test.Config.get_nested") as get_nested:
            get_nested.return_value = False
            solver.createCoelhoValidInequalities()

        assert not list(solver.model.iter_constraints())

    def test_coelho_inequalities_add_logical_constraints(self):
        module = _load_real_mpprp()
        solver = module.MultProductProdctionRoutingProblem(
            map=self._data(num_customers=2, num_vehicles=3),
            dir="/tmp",
            log=DummyLogger(),
            start={"start": False},
            coelho_inequalities=True,
        )
        solver.createDecisionVariables()
        solver.createCoelhoValidInequalities()

        names = [constraint.name for constraint in solver.model.iter_constraints()]

        assert sum(name.startswith("COELHO_15_") for name in names) == 6
        assert sum(name.startswith("COELHO_16_") for name in names) == 18
        assert sum(name.startswith("COELHO_17_") for name in names) == 6

    def test_solver_node_count_uses_cplex_progress_when_details_are_zero(self):
        module = _load_real_mpprp()
        solver = module.MultProductProdctionRoutingProblem(
            map=self._data(),
            dir="/tmp",
            log=DummyLogger(),
            start={"start": False},
        )

        class Progress:
            def get_num_nodes_processed(self):
                return 17

            def get_num_nodes_remaining(self):
                return 4

        solver.model = SimpleNamespace(
            solve_details=SimpleNamespace(best_bound=12.5, nb_nodes_processed=0),
            solution=object(),
            get_cplex=lambda: SimpleNamespace(
                solution=SimpleNamespace(progress=Progress())
            ),
        )

        solver.processInformationsSolver()

        assert solver.nodeCount == 17
        assert solver.nodesRemaining == 4

    def test_node_progress_keeps_largest_callback_count(self):
        module = _load_real_mpprp()
        solver = module.MultProductProdctionRoutingProblem(
            map=self._data(),
            dir="/tmp",
            log=DummyLogger(),
            start={"start": False},
        )

        solver._update_node_progress(processed=63, remaining=5)
        solver._update_node_progress(processed=17, remaining=2)

        assert solver.nodeCount == 63
        assert solver.nodesRemaining == 2

    def test_bound_progress_records_root_evolution(self):
        module = _load_real_mpprp()
        solver = module.MultProductProdctionRoutingProblem(
            map=self._data(),
            dir="/tmp",
            log=DummyLogger(),
            start={"start": False},
        )
        solver.telemetry_config = {
            "enabled": True,
            "gap_target_relative": 0.01,
            "bound_progress_interval_seconds": 1.0,
        }

        solver._record_bound_progress(
            1.0,
            0.043,
            incumbent_objective=0.764,
            relative_gap=0.944,
            nodes_processed=0,
            nodes_remaining=1,
        )
        solver._record_bound_progress(
            1.2,
            0.043,
            incumbent_objective=0.764,
            relative_gap=0.944,
            nodes_processed=0,
            nodes_remaining=1,
        )

        events = [event for event in solver._telemetry_events if event["event"] == "bound_progress"]
        assert len(events) == 1
        assert events[0]["best_bound"] == 0.043
        assert events[0]["is_root"] is True
        assert solver.get_telemetry()["root_bound"] == 0.043

    def test_rounded_capacity_schedule_separates_root_and_spaced_early_nodes(self):
        module = _load_real_mpprp()
        solver = module.MultProductProdctionRoutingProblem(
            map=self._data(),
            dir="/tmp",
            log=DummyLogger(),
            start={"start": False},
        )
        solver.rounded_capacity_config = {
            "enabled": True,
            "node_frequency": 5,
            "max_non_root_node": 20,
        }

        assert solver._rounded_capacity_should_separate(0) is True
        assert solver._rounded_capacity_should_separate(5) is True
        assert solver._rounded_capacity_should_separate(6) is False
        assert solver._rounded_capacity_should_separate(25) is False

    def test_rounded_capacity_row_maps_customer_subset_to_model_nodes(self):
        module = _load_real_mpprp()
        solver = module.MultProductProdctionRoutingProblem(
            map=self._data(num_customers=2),
            dir="/tmp",
            log=DummyLogger(),
            start={"start": False},
        )
        solver.createDecisionVariables()
        solver.rounded_capacity_config = {"enabled": True}
        solver._prepare_rounded_capacity_callback_data()

        indices, values = solver._rounded_capacity_build_row(
            RoundedCapacityCut(tau=0, customers=(0,), rhs=2, violation=1.0)
        )

        assert len(indices) == 4
        assert values == [1.0] * 4

    def test_rounded_capacity_disabled_by_default_does_not_register_callback(self):
        module = _load_real_mpprp()
        solver = module.MultProductProdctionRoutingProblem(
            map=self._data(),
            dir="/tmp",
            log=DummyLogger(),
            start={"start": False},
        )

        assert solver._install_rounded_capacity_callback() is False

    def test_rounded_capacity_callback_materializes_cut_and_deduplicates(self):
        if importlib.util.find_spec("cplex") is None:
            self.skipTest("CPLEX não está instalado neste ambiente")
        module = _load_real_mpprp()
        solver = module.MultProductProdctionRoutingProblem(
            map=self._data(),
            dir="/tmp",
            log=DummyLogger(),
            start={"start": False},
        )
        solver.rounded_capacity_config = {"enabled": True, "min_violation": 1e-6}
        solver.createDecisionVariables()
        solver._prepare_rounded_capacity_callback_data()

        route_indices = {
            index
            for t, v, i, k, index in solver._rounded_capacity_z_indices
            if t == 0 and i == 0 and k == 1 or t == 0 and i == 1 and k == 0
        }

        class Callback:
            def __init__(self):
                self.added = []

            def get_values(self, indices):
                return [0.5 if index in route_indices else 0.0 for index in indices]

            def add(self, pair, sense, rhs):
                self.added.append((pair, sense, rhs))

        callback = Callback()
        solver._run_rounded_capacity_separator(callback, max_cuts=3)
        solver._run_rounded_capacity_separator(callback, max_cuts=3)

        assert len(callback.added) == 1
        assert callback.added[0][1:] == ("G", 2)
        assert len(callback.added[0][0].ind) == 2
        assert solver._rounded_capacity_stats["duplicate_cuts"] == 1


if __name__ == "__main__":
    unittest.main()
