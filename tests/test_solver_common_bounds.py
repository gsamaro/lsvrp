import numpy as np

from src.solvers._solver_common import (
    ProblemData,
    compute_delivery_upper_bounds,
    compute_inventory_upper_bounds,
)


def _problem_data():
    return ProblemData.from_map(
        {
            "num_products": 2,
            "num_customers": 2,
            "num_periods": 3,
            "num_vehicles": 4,
            "B": 100,
            "b_p": [1, 1],
            "c_p": [1, 1],
            "s_p": [1, 1],
            "M": 100,
            "U_pi": [[50, 10, 2], [50, 4, 100]],
            "I_pi0": [[7, 2, 1], [9, 0, 10]],
            "h_pi": [[0, 1, 1], [0, 1, 1]],
            "C": 6,
            "f": 1,
            "a_ik": [[0, 1, 1], [1, 0, 1], [1, 1, 0]],
            "d_pit": [
                [[3, 4, 2], [1, 2, 1]],
                [[5, 1, 3], [2, 4, 1]],
            ],
        }
    )


def test_customer_inventory_bounds_subtract_cumulative_demand_without_vehicle_factor():
    problem = _problem_data()

    bounds = compute_inventory_upper_bounds(problem)

    # Product 0, customer 1: qbar is 6 in each period; net receipts are
    # (6-3, 6-4, 6-2), added to initial stock 2 and capped at U=10.
    np.testing.assert_allclose(bounds[0, 1, :], [5, 7, 10])
    # Four vehicles are available, but visit uniqueness permits only one
    # vehicle to deliver to this customer in any given period.
    assert bounds[0, 1, 0] == 5


def test_customer_inventory_bounds_cover_multiple_products_and_capacity_caps():
    bounds = compute_inventory_upper_bounds(_problem_data())

    np.testing.assert_allclose(bounds[0, 2, :], [2, 2, 2])
    np.testing.assert_allclose(bounds[1, 1, :], [1, 4, 4])
    np.testing.assert_allclose(bounds[1, 2, :], [14, 16, 21])


def test_delivery_upper_bounds_use_vehicle_capacity_and_customer_capacity_plus_demand():
    delivery_bounds = compute_delivery_upper_bounds(_problem_data())

    np.testing.assert_allclose(delivery_bounds[0, 1, :], [6, 6, 6])
    np.testing.assert_allclose(delivery_bounds[0, 2, :], [3, 4, 3])
    np.testing.assert_allclose(delivery_bounds[1, 1, :], [6, 5, 6])


def test_plant_inventory_bound_is_unchanged():
    bounds = compute_inventory_upper_bounds(_problem_data())

    # The plant bound remains initial inventory plus cumulative production UB.
    np.testing.assert_allclose(bounds[0, 0, :], [20, 29, 32])
    np.testing.assert_allclose(bounds[1, 0, :], [25, 34, 38])


def test_customer_inventory_bound_is_clamped_to_nonnegative():
    data = _problem_data().raw_map.copy()
    data["num_customers"] = 1
    data["num_periods"] = 2
    data["num_vehicles"] = 2
    data["U_pi"] = [[20, 20], [20, 20]]
    data["I_pi0"] = [[0, 0], [0, 0]]
    data["h_pi"] = [[0, 1], [0, 1]]
    data["d_pit"] = [[[8, 0]], [[8, 0]]]
    data["a_ik"] = [[0, 1], [1, 0]]
    problem = ProblemData.from_map(data)

    bounds = compute_inventory_upper_bounds(problem)

    assert np.all(bounds[:, 1, :] >= 0)
    np.testing.assert_allclose(bounds[:, 1, 0], [0, 0])
