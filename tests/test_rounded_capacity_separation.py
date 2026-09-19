import numpy as np
import pytest

from src.solvers.RoundedCapacitySeparation import (
    aggregate_route_arcs,
    build_obligatory_demands,
    cut_value,
    separate_cumulative_cuts,
)


def test_obligatory_demand_keeps_products_separate_and_uses_customer_stock():
    demand = np.array(
        [
            [[3, 0, 2], [0, 4, 0]],
            [[0, 1, 0], [2, 0, 3]],
        ],
        dtype=float,
    )
    initial_inventory = np.array([[0, 2, 0], [0, 0, 1]], dtype=float)

    result = build_obligatory_demands(demand, initial_inventory)

    np.testing.assert_allclose(result, [[1, 1], [2, 5], [4, 8]])


def test_aggregate_route_arcs_counts_oriented_arc_pair_once():
    z_values = np.zeros((1, 1, 3, 3))
    z_values[0, 0, 0, 1] = 0.25
    z_values[0, 0, 1, 0] = 0.5

    aggregated = aggregate_route_arcs(z_values, tau=0)

    assert aggregated[0, 1] == pytest.approx(0.75)
    assert cut_value(aggregated, [1]) == pytest.approx(0.75)


def test_separator_uses_customer_nodes_after_depot_and_finds_unit_subset():
    z_values = np.zeros((1, 1, 3, 3))
    z_values[0, 0, 0, 1] = 0.5
    z_values[0, 0, 1, 0] = 0.5
    obligatory = np.array([[6.0, 11.0]])

    cuts = separate_cumulative_cuts(
        z_values,
        obligatory,
        capacity=10,
        max_cuts=2,
        customer_node_offset=1,
    )

    assert cuts
    assert cuts[0].customers == (1,)
    assert cuts[0].rhs == 4
    assert cuts[0].violation == pytest.approx(4)


def test_separator_uses_cumulative_prefix_and_limits_output():
    z_values = np.zeros((2, 1, 2, 2))
    z_values[0, 0, 0, 1] = 0.5
    z_values[0, 0, 1, 0] = 0.5
    obligatory = np.array([[6.0], [11.0]])

    cuts = separate_cumulative_cuts(
        z_values,
        obligatory,
        capacity=10,
        max_cuts=1,
        customer_node_offset=0,
    )

    assert len(cuts) == 1
    assert cuts[0].tau == 1
    assert cuts[0].rhs == 4
    assert cuts[0].violation == pytest.approx(3)


def test_separator_has_no_violation_for_a_complete_two_customer_route():
    z_values = np.zeros((1, 1, 3, 3))
    for customer in (1, 2):
        z_values[0, 0, 0, customer] = 1.0
        z_values[0, 0, customer, 0] = 1.0
    obligatory = np.array([[5.0, 5.0]])

    cuts = separate_cumulative_cuts(
        z_values, obligatory, capacity=10, customer_node_offset=1
    )

    assert cuts == []


def test_separator_respects_rounding_tolerance_and_deduplicates():
    z_values = np.zeros((1, 1, 2, 2))
    z_values[0, 0, 0, 1] = 0.5
    z_values[0, 0, 1, 0] = 0.5
    obligatory = np.array([[10.0 + 1e-10]])
    statistics = {}

    cuts = separate_cumulative_cuts(
        z_values,
        obligatory,
        capacity=10,
        max_cuts=10,
        min_violation=0.1,
        tolerance=1e-9,
        statistics=statistics,
    )

    assert len(cuts) == 1
    assert cuts[0].rhs == 2
    assert statistics["seeds"] == 1
    assert statistics["candidate_evaluations"] > 0


def test_separator_rejects_incompatible_customer_dimensions():
    with pytest.raises(ValueError, match="todos os nos"):
        separate_cumulative_cuts(
            np.zeros((1, 1, 2, 2)),
            np.array([[1.0, 1.0]]),
            capacity=10,
            customer_node_offset=1,
        )
