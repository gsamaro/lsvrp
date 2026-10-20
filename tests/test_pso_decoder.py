"""PSO decoder options: setup threshold, Clarke-Wright partition and route improvement.

The Python reference and the numba kernel must agree for every option, and the
default options of ``FeasibleParticleHeuristic`` must keep the legacy decoder.
"""

from itertools import product
from unittest.mock import Mock

import numpy as np
import pytest

from config import Config
from src.solvers.FeasibleParticleHeuristic import FeasibleParticleHeuristic
from src.solvers.ParticleSwarmOptimization import ParticleSwarmOptimization

IMPROVED_DECODER = {
    "setup_threshold": 0.5,
    "routing": "savings_2opt",
    "delivery": "lookahead",
    "delivery_threshold": 0.5,
    "max_cover": 2,
    "fleet_utilization": 0.9,
}

STATE_FIELDS = (
    "X",
    "Y",
    "I",
    "Q",
    "assignments",
    "route_nodes",
    "route_lengths",
    "feasible",
)


def _data(customers=8, products=3, vehicles=4, periods=5, seed=0, capacity=40):
    rng = np.random.default_rng(seed)
    nodes = customers + 1
    xy = rng.integers(0, 30, size=(nodes, 2))
    distance = np.rint(np.linalg.norm(xy[:, None] - xy[None], axis=2)).astype(int)
    demand = rng.integers(0, 6, size=(products, customers, periods))
    return {
        "file": "unit.dat",
        "num_products": products,
        "num_customers": customers,
        "num_periods": periods,
        "num_vehicles": vehicles,
        "B": 400,
        "b_p": [1, 2, 1][:products],
        "c_p": [2, 3, 1][:products],
        "s_p": [30, 50, 20][:products],
        "M": 1000,
        "U_pi": [[200] + [60] * customers for _ in range(products)],
        "I_pi0": [
            [int(rng.integers(0, 12))]
            + [int(rng.integers(0, 6)) for _ in range(customers)]
            for _ in range(products)
        ],
        "h_pi": [[1] + [2] * customers for _ in range(products)],
        "C": capacity,
        "f": 25,
        "a_ik": distance.tolist(),
        "d_pit": demand.tolist(),
        "weight": [0.2] * 5,
        "alpha": 0.5,
    }


def _heuristic(data, backend, decoder=None, bounds=None):
    return FeasibleParticleHeuristic(
        data,
        "/tmp",
        Mock(),
        execution_backend=backend,
        parallel_workers=1,
        bounds=bounds,
        decoder=decoder,
    )


def _bounds(data, seed=3, with_delivery=False):
    rng = np.random.default_rng(seed)
    shape = (data["num_products"], data["num_periods"])
    lower = rng.integers(0, 15, size=shape).astype(float)
    bounds = {"lower": lower, "upper": lower + rng.integers(1, 40, size=shape)}
    if with_delivery:
        q_shape = (
            data["num_products"],
            data["num_vehicles"],
            data["num_customers"] + 1,
            data["num_periods"],
        )
        q_lower = rng.integers(0, 3, size=q_shape).astype(float)
        bounds["q_lower"] = q_lower
        bounds["q_upper"] = q_lower + rng.integers(1, 6, size=q_shape)
    return bounds


FLOAT_FIELDS = ("X", "I", "Q")


def _assert_same_states(expected, actual, exact=True):
    """Integer fields always match exactly; floats do too unless ``exact`` is off.

    The proportional load levelling multiplies and accumulates, and the compiled
    kernel may fuse those operations, so quantities can differ by one ulp.
    """
    for name in STATE_FIELDS:
        if exact or name not in FLOAT_FIELDS:
            np.testing.assert_array_equal(
                getattr(actual, name), getattr(expected, name), err_msg=name
            )
        else:
            np.testing.assert_allclose(
                getattr(actual, name),
                getattr(expected, name),
                rtol=1e-12,
                atol=1e-9,
                err_msg=name,
            )
    np.testing.assert_allclose(actual.costs, expected.costs, rtol=1e-12)


# ------------------------------------------------------------------ option checks


@pytest.mark.parametrize(
    "decoder",
    [
        {"setup_threshold": 1.0},
        {"setup_threshold": -0.1},
        {"setup_threshold": True},
        {"setup_threshold": "0.5"},
        {"routing": "tsp"},
    ],
)
def test_invalid_decoder_options_are_rejected(decoder):
    with pytest.raises(ValueError, match="decoder"):
        _heuristic(_data(), "python", decoder)


def test_defaults_keep_the_legacy_decoder():
    heuristic = _heuristic(_data(), "python")
    assert heuristic.decoder == {
        "setup_threshold": 0.0,
        "routing": "nearest_neighbor",
        "delivery": "just_in_time",
        "delivery_threshold": 0.5,
        "max_cover": 2,
        "fleet_utilization": 0.9,
    }


def test_asymmetric_distances_fall_back_to_nearest_neighbor():
    data = _data()
    data["a_ik"][1][2] += 5
    log = Mock()
    heuristic = FeasibleParticleHeuristic(
        data, "/tmp", log, decoder={"routing": "savings_2opt"}
    )
    assert heuristic.decoder["routing"] == "nearest_neighbor"
    log.warning.assert_called_once()


# ------------------------------------------------- numba / python reference parity


@pytest.mark.parametrize(
    ("threshold", "routing", "with_bounds"),
    list(
        product(
            [0.0, 0.5],
            ["nearest_neighbor", "savings_2opt"],
            [False, True],
        )
    ),
)
def test_numba_backend_matches_python_reference(threshold, routing, with_bounds):
    data = _data(seed=11)
    decoder = {"setup_threshold": threshold, "routing": routing}
    bounds = _bounds(data) if with_bounds else None
    python = _heuristic(data, "python", decoder, bounds)
    numba = _heuristic(data, "numba", decoder, bounds)
    positions = np.random.default_rng(5).normal(size=(40, python.particle_dim))

    expected = python.build_population_state(positions)
    actual = numba.build_population_state(positions)

    assert expected.feasible.any()
    _assert_same_states(expected, actual)


@pytest.mark.parametrize("routing", ["nearest_neighbor", "savings_2opt"])
def test_numba_backend_matches_python_with_delivery_bounds(routing):
    # Delivery bounds steer the drain of plant stock, which runs after the partition.
    data = _data(seed=21)
    decoder = {"setup_threshold": 0.5, "routing": routing}
    bounds = _bounds(data, with_delivery=True)
    python = _heuristic(data, "python", decoder, bounds)
    numba = _heuristic(data, "numba", decoder, bounds)
    positions = np.random.default_rng(8).normal(size=(30, python.particle_dim))
    expected = python.build_population_state(positions)
    assert expected.feasible.any()
    _assert_same_states(expected, numba.build_population_state(positions))


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_numba_backend_matches_python_on_tight_fleets(seed):
    # Few vehicles with large loads exercise the best-fit fallback of the partition.
    data = _data(customers=9, vehicles=3, capacity=28, seed=seed)
    decoder = {"setup_threshold": 0.5, "routing": "savings_2opt"}
    python = _heuristic(data, "python", decoder)
    numba = _heuristic(data, "numba", decoder)
    positions = np.random.default_rng(seed).normal(size=(40, python.particle_dim))
    _assert_same_states(
        python.build_population_state(positions),
        numba.build_population_state(positions),
    )


# ---------------------------------------------------------------- setup threshold


def _two_period_data():
    data = {
        "file": "unit.dat",
        "num_products": 1,
        "num_customers": 1,
        "num_periods": 2,
        "num_vehicles": 1,
        "B": 100,
        "b_p": [1],
        "c_p": [1],
        "s_p": [10],
        "M": 1000,
        "U_pi": [[50, 50]],
        # The plant stock covers the whole demand of period 0.
        "I_pi0": [[8, 0]],
        "h_pi": [[0, 1]],
        "C": 100,
        "f": 5,
        "a_ik": [[0, 4], [4, 0]],
        "d_pit": [[[3, 3]]],
        "weight": [0.2] * 5,
        "alpha": 0.5,
    }
    return data


@pytest.mark.parametrize("backend", ["python", "numba"])
@pytest.mark.parametrize("gene", [-2.0, -30.0])
def test_threshold_lets_a_period_skip_its_setup(backend, gene):
    data = _two_period_data()
    # Both genes give a sigmoid at most the threshold: no extra production wanted.
    particle = np.full((1, 2 + 1 * 1 * 2 * 2), gene)

    legacy = _heuristic(data, backend).build_population_state(particle)
    skipping = _heuristic(
        data, backend, {"setup_threshold": 0.5}
    ).build_population_state(particle)

    assert bool(legacy.feasible[0]) and bool(skipping.feasible[0])
    # Legacy: any gene adds some production, so period 0 pays a setup even when the
    # plant stock already covers the demand (a tiny positive quantity still counts).
    assert legacy.Y[0, 0, 0] == 1
    assert legacy.X[0, 0, 0] > 0
    # Threshold: period 0 is served from stock and has no setup at all.
    assert skipping.Y[0, 0, 0] == 0
    assert skipping.X[0, 0, 0] == 0
    assert skipping.costs[0] < legacy.costs[0]


@pytest.mark.parametrize("backend", ["python", "numba"])
def test_genes_above_the_threshold_still_produce_ahead(backend):
    data = _two_period_data()
    particle = np.full((1, 2 + 4), 8.0)
    state = _heuristic(data, backend, {"setup_threshold": 0.5}).build_population_state(
        particle
    )
    assert bool(state.feasible[0])
    assert state.X[0, 0, 0] > 1.0
    assert state.Y[0, 0, 0] == 1


# ------------------------------------------------------------ Clarke-Wright partition


def _clusters_data():
    """Two geographic clusters; best-fit by load pairs customers across them."""
    plant, near_a, near_b = (0, 0), [(10, 0), (11, 0)], [(0, 10), (0, 11)]
    points = [plant, near_a[0], near_b[0], near_b[1], near_a[1]]
    distance = np.rint(
        np.linalg.norm(np.array(points)[:, None] - np.array(points)[None], axis=2)
    ).astype(int)
    # Customers 1 and 4 sit in cluster A, 2 and 3 in cluster B.
    demand = [[[6], [4], [6], [4]]]
    return {
        "file": "unit.dat",
        "num_products": 1,
        "num_customers": 4,
        "num_periods": 1,
        "num_vehicles": 2,
        "B": 1000,
        "b_p": [1],
        "c_p": [1],
        "s_p": [1],
        "M": 1000,
        "U_pi": [[100] * 5],
        "I_pi0": [[0] * 5],
        "h_pi": [[0] * 5],
        "C": 10,
        "f": 1,
        "a_ik": distance.tolist(),
        "d_pit": demand,
        "weight": [0.2] * 5,
        "alpha": 0.5,
    }


@pytest.mark.parametrize("backend", ["python", "numba"])
def test_savings_groups_neighbours_where_best_fit_mixes_clusters(backend):
    data = _clusters_data()
    particle = np.zeros((1, 1 * 1 + 1 * 2 * 5 * 1))

    legacy = _heuristic(data, backend).build_population(particle)[0]
    savings = _heuristic(data, backend, {"routing": "savings_2opt"}).build_population(
        particle
    )[0]

    legacy_sets = sorted(sorted(route) for route in legacy["route_plan"][0] if route)
    savings_sets = sorted(sorted(route) for route in savings["route_plan"][0] if route)
    assert legacy_sets == [[1, 2], [3, 4]]
    assert savings_sets == [[1, 4], [2, 3]]
    assert savings["cost"] < legacy["cost"]


def _fallback_data():
    """Savings builds three routes for two vehicles; best-fit packs them in two."""
    data = _clusters_data()
    far, close = 15, 1
    matrix = [[0 if i == k else far for k in range(5)] for i in range(5)]
    for k in range(1, 5):
        matrix[0][k] = matrix[k][0] = 10
    matrix[2][3] = matrix[3][2] = close
    data["a_ik"] = matrix
    # Loads 6, 4, 5, 5 with C = 10: merging customers 2 and 3 (9) strands 6 and 5.
    data["d_pit"] = [[[6], [4], [5], [5]]]
    return data


@pytest.mark.parametrize("backend", ["python", "numba"])
def test_partition_falls_back_to_best_fit_above_fleet_size(backend):
    data = _fallback_data()
    particle = np.zeros((1, 1 + 2 * 5))
    solution = _heuristic(data, backend, {"routing": "savings_2opt"}).build_population(
        particle
    )[0]
    routes = [route for route in solution["route_plan"][0] if route]
    assert solution["feasible"]
    assert len(routes) == 2
    assert sorted(customer for route in routes for customer in route) == [1, 2, 3, 4]


# ---------------------------------------------------------------- route improvement


def _route_cost(heuristic, route):
    nodes = [0, *route, 0]
    return sum(heuristic._distance[a, b] for a, b in zip(nodes, nodes[1:]))


def test_improve_route_removes_crossings_and_never_worsens():
    data = _data(seed=4)
    heuristic = _heuristic(data, "python", {"routing": "savings_2opt"})
    rng = np.random.default_rng(0)
    improved_any = False
    for _ in range(30):
        route = rng.permutation(np.arange(1, 8)).tolist()
        better = heuristic._improve_route(route)
        assert sorted(better) == sorted(route)
        assert _route_cost(heuristic, better) <= _route_cost(heuristic, route) + 1e-9
        improved_any |= _route_cost(heuristic, better) < _route_cost(heuristic, route)
        # A local optimum is a fixed point.
        assert heuristic._improve_route(better) == better
    assert improved_any


@pytest.mark.parametrize("backend", ["python", "numba"])
def test_savings_routes_are_not_longer_than_nearest_neighbor_routes(backend):
    data = _data(customers=10, vehicles=5, seed=6, capacity=60)
    particles = np.random.default_rng(1).normal(size=(60, 3 * 5 + 3 * 5 * 11 * 5))
    legacy = _heuristic(data, backend).build_population_state(particles)
    savings = _heuristic(
        data, backend, {"routing": "savings_2opt"}
    ).build_population_state(particles)
    both = np.flatnonzero(legacy.feasible & savings.feasible)
    assert len(both)
    # Same production decisions, so the objective difference is routing only.
    assert np.mean(savings.costs[both]) < np.mean(legacy.costs[both])


# ----------------------------------------------------------------------- PSO wiring


@pytest.fixture
def pso_config(monkeypatch):
    value = Config.normalize({"solver": {"multiobjective": False, "method": "PSO"}})
    monkeypatch.setattr(Config, "_data", value)
    return value


def _pso(data, config):
    config["solver"]["pso"].update(
        {
            "lot_sizing_bounds": {"enabled": False},
            "execution_backend": "python",
            "jit_warmup": False,
            "swarm_size": 2,
            "max_iterations": 0,
        }
    )
    return ParticleSwarmOptimization(data, "/tmp", Mock())


def test_pso_uses_the_improved_decoder_by_default(pso_config):
    pso = _pso(_data(customers=3, products=1, vehicles=2, periods=2), pso_config)
    assert pso.pso_config["decoder"] == IMPROVED_DECODER
    assert pso.heuristic.decoder == pso.pso_config["decoder"]


def test_pso_decoder_options_can_restore_the_legacy_decoder(pso_config):
    legacy = {
        "setup_threshold": 0.0,
        "routing": "nearest_neighbor",
        "delivery": "just_in_time",
    }
    pso_config["solver"]["pso"]["decoder"] = legacy
    pso = _pso(_data(customers=3, products=1, vehicles=2, periods=2), pso_config)
    assert {k: pso.heuristic.decoder[k] for k in legacy} == legacy


def test_pso_partial_decoder_override_keeps_the_other_default(pso_config):
    pso_config["solver"]["pso"]["decoder"] = {"routing": "nearest_neighbor"}
    pso = _pso(_data(customers=3, products=1, vehicles=2, periods=2), pso_config)
    assert pso.heuristic.decoder == {**IMPROVED_DECODER, "routing": "nearest_neighbor"}


# ------------------------------------------------------------------------- config


def test_config_defaults_for_the_decoder():
    decoder = Config.normalize({})["solver"]["pso"]["decoder"]
    assert decoder == IMPROVED_DECODER


def test_config_keeps_a_partial_decoder_block_and_fills_the_rest():
    configured = Config.normalize(
        {"solver": {"pso": {"decoder": {"setup_threshold": 0.25}}}}
    )
    assert configured["solver"]["pso"]["decoder"] == {
        **IMPROVED_DECODER,
        "setup_threshold": 0.25,
    }


@pytest.mark.parametrize(
    ("decoder", "message"),
    [
        ({"setup_threshold": 1}, "setup_threshold"),
        ({"setup_threshold": -0.5}, "setup_threshold"),
        ({"setup_threshold": "0.5"}, "setup_threshold"),
        ({"setup_threshold": True}, "setup_threshold"),
        ({"routing": "tsp"}, "routing"),
    ],
)
def test_config_rejects_invalid_decoder_options(decoder, message):
    with pytest.raises(ValueError, match=message):
        Config.normalize({"solver": {"pso": {"decoder": decoder}}})


# ------------------------------------------------------------ delivery lookahead

LOOKAHEAD = {
    "setup_threshold": 0.5,
    "routing": "savings_2opt",
    "delivery": "lookahead",
}


def _single_customer(
    periods=4, capacity=100, stock=50, production=1000, demand=3, vehicles=1
):
    return {
        "file": "unit.dat",
        "num_products": 1,
        "num_customers": 1,
        "num_periods": periods,
        "num_vehicles": vehicles,
        "B": production,
        "b_p": [1],
        "c_p": [1],
        "s_p": [10],
        "M": 1000,
        "U_pi": [[500, stock]],
        "I_pi0": [[0, 0]],
        "h_pi": [[0, 1]],
        "C": capacity,
        "f": 5,
        "a_ik": [[0, 4], [4, 0]],
        "d_pit": [[[demand] * periods]],
        "weight": [0.2] * 5,
        "alpha": 0.5,
    }


def _cover_particle(heuristic, cover_period=0, gene=10.0):
    """All genes off except the cover gene of customer 1 at ``cover_period``."""
    particle = np.full((1, heuristic.particle_dim), -10.0)
    particle[0, heuristic.delivery_gene_indices[cover_period]] = gene
    return particle


def _tight_peak_data():
    """Periods 2 and 3 carry 100 units for a fleet of 90: just in time is invalid."""
    data = _data(customers=4, products=1, vehicles=2, periods=4, capacity=45, seed=0)
    data["B"], data["U_pi"] = 1000, [[500] + [60] * 4]
    data["I_pi0"] = [[0] * 5]
    data["a_ik"] = [[0 if i == k else 10 + i + k for k in range(5)] for i in range(5)]
    data["d_pit"] = [[[5, 5, 25, 25] for _ in range(4)]]
    return data


def test_delivery_genes_index_one_gene_per_customer_and_period():
    heuristic = _heuristic(_data(customers=3, vehicles=2, periods=4), "python")
    indices = heuristic.delivery_gene_indices
    assert len(indices) == 3 * 4
    assert len(set(indices.tolist())) == len(indices)
    assert indices.min() >= heuristic.dim_x
    assert indices.max() < heuristic.particle_dim


@pytest.mark.parametrize(
    "decoder",
    [
        {"delivery": "tomorrow"},
        {"delivery_threshold": 1.0},
        {"delivery_threshold": True},
        {"max_cover": 0},
        {"max_cover": 1.5},
        {"max_cover": True},
        {"fleet_utilization": 0.0},
        {"fleet_utilization": 1.5},
        {"fleet_utilization": "0.9"},
    ],
)
def test_invalid_delivery_options_are_rejected(decoder):
    with pytest.raises(ValueError, match="decoder"):
        _heuristic(_data(), "python", decoder)


@pytest.mark.parametrize("routing", ["nearest_neighbor", "savings_2opt"])
@pytest.mark.parametrize("with_bounds", [False, True])
@pytest.mark.parametrize("builder", ["loose", "tight", "single"])
def test_numba_backend_matches_python_for_lookahead(routing, with_bounds, builder):
    data = {
        "loose": lambda: _data(seed=31),
        "tight": _tight_peak_data,
        "single": _single_customer,
    }[builder]()
    decoder = {**LOOKAHEAD, "routing": routing}
    bounds = _bounds(data) if with_bounds else None
    python = _heuristic(data, "python", decoder, bounds)
    numba = _heuristic(data, "numba", decoder, bounds)
    positions = np.random.default_rng(13).normal(size=(40, python.particle_dim))
    expected = python.build_population_state(positions)
    _assert_same_states(expected, numba.build_population_state(positions), exact=False)
    if not with_bounds:
        assert expected.feasible.any()


@pytest.mark.parametrize("backend", ["python", "numba"])
def test_lookahead_without_delivery_genes_equals_just_in_time(backend):
    data = _data(seed=41)
    jit = _heuristic(data, backend, {**LOOKAHEAD, "delivery": "just_in_time"})
    ahead = _heuristic(data, backend, LOOKAHEAD)
    assert not ahead._guard_active
    positions = np.random.default_rng(2).normal(size=(30, jit.particle_dim))
    positions[:, ahead.delivery_gene_indices] = -10.0
    _assert_same_states(
        jit.build_population_state(positions), ahead.build_population_state(positions)
    )


@pytest.mark.parametrize("backend", ["python", "numba"])
def test_load_levelling_makes_a_just_in_time_infeasible_instance_feasible(backend):
    data = _tight_peak_data()
    jit = _heuristic(data, backend, {**LOOKAHEAD, "delivery": "just_in_time"})
    ahead = _heuristic(data, backend, LOOKAHEAD)
    assert ahead._guard_active
    random = np.random.default_rng(0).normal(size=(50, ahead.particle_dim))
    # With the delivery genes off the levelling alone must make every particle valid.
    positions = random.copy()
    positions[:, ahead.delivery_gene_indices] = -10.0

    assert not jit.build_population_state(random).feasible.any()
    # Random delivery genes may add unpackable cover, but plenty stays valid.
    assert ahead.build_population_state(random).feasible.mean() > 0.3
    state = ahead.build_population_state(positions)
    assert state.feasible.all()
    for index in range(len(state)):
        solution = ahead.solution_from_state(state, index)
        full = ahead.materialize_solution(solution)
        assert ahead.validate_solution(full)["feasible"]
        loads = state.Q[index].sum(axis=(0, 1, 2))
        assert loads.max() <= ahead._fleet_cap + 1e-6


@pytest.mark.parametrize("backend", ["python", "numba"])
def test_levelling_moves_only_the_overflow_to_the_latest_possible_period(backend):
    data = _tight_peak_data()
    heuristic = _heuristic(data, backend, LOOKAHEAD)
    particle = np.full((1, heuristic.particle_dim), -10.0)
    state = heuristic.build_population_state(particle)
    assert bool(state.feasible[0])
    loads = state.Q[0].sum(axis=(0, 1, 2))
    # 81 = 0.9 * 90 usable units: periods 2 and 3 are cut to 81 and the 38 units of
    # overflow are shipped at the last moments (period 1 and period 2).
    np.testing.assert_allclose(loads, [20, 58, 81, 81])
    np.testing.assert_allclose(loads.sum(), 240)
    assert state.I[0, :, :, -1].sum() == pytest.approx(0)


@pytest.mark.parametrize("backend", ["python", "numba"])
def test_cover_gene_consolidates_visits_and_setups(backend):
    data = _single_customer()
    heuristic = _heuristic(data, backend, LOOKAHEAD)
    state = heuristic.build_population_state(_cover_particle(heuristic))

    assert bool(state.feasible[0])
    # t0 delivers its deficit (3) plus the next two periods (6).
    np.testing.assert_allclose(state.Q[0, 0, 0, 1], [9, 0, 0, 3])
    np.testing.assert_allclose(state.X[0, 0], [9, 0, 0, 3])
    np.testing.assert_array_equal(state.Y[0, 0], [1, 0, 0, 1])
    np.testing.assert_array_equal(state.route_lengths[0, :, 0], [1, 0, 0, 1])
    jit = _heuristic(data, backend, {**LOOKAHEAD, "delivery": "just_in_time"})
    just_in_time = jit.build_population_state(_cover_particle(jit))
    np.testing.assert_array_equal(just_in_time.route_lengths[0, :, 0], [1, 1, 1, 1])
    assert state.costs[0] != just_in_time.costs[0]


@pytest.mark.parametrize("backend", ["python", "numba"])
def test_cover_is_clipped_by_customer_storage(backend):
    heuristic = _heuristic(_single_customer(stock=5), backend, LOOKAHEAD)
    state = heuristic.build_population_state(_cover_particle(heuristic))
    assert bool(state.feasible[0])
    assert state.Q[0, 0, 0, 1, 0] == pytest.approx(8)  # deficit 3 + 5 of storage
    assert state.I[0, 0, 1, 0] == pytest.approx(5)


@pytest.mark.parametrize("backend", ["python", "numba"])
def test_cover_is_clipped_by_vehicle_capacity(backend):
    # Two vehicles keep the usable fleet capacity (12.6) above C, so C is what binds.
    heuristic = _heuristic(_single_customer(capacity=7, vehicles=2), backend, LOOKAHEAD)
    state = heuristic.build_population_state(_cover_particle(heuristic))
    assert bool(state.feasible[0])
    assert state.Q[0, 0, 0, 1, 0] == pytest.approx(7)


@pytest.mark.parametrize("backend", ["python", "numba"])
def test_cover_is_clipped_by_usable_fleet_capacity(backend):
    heuristic = _heuristic(_single_customer(capacity=8), backend, LOOKAHEAD)
    state = heuristic.build_population_state(_cover_particle(heuristic))
    assert bool(state.feasible[0])
    assert state.Q[0, 0, 0, 1, 0] == pytest.approx(0.9 * 8)


@pytest.mark.parametrize("backend", ["python", "numba"])
def test_cover_is_clipped_by_production_time(backend):
    # Only 5 time units per period: the 6 units ahead are scaled to what fits.
    heuristic = _heuristic(_single_customer(production=5), backend, LOOKAHEAD)
    state = heuristic.build_population_state(_cover_particle(heuristic))
    assert bool(state.feasible[0])
    assert state.X[0, 0, 0] <= 5 + 1e-8
    assert state.X[0, 0, 0] == pytest.approx(5)


def test_threshold_gates_the_delivery_gene():
    heuristic = _heuristic(_single_customer(), "python", LOOKAHEAD)
    below = heuristic.build_population_state(_cover_particle(heuristic, gene=-0.01))
    above = heuristic.build_population_state(_cover_particle(heuristic, gene=0.4))
    assert below.Q[0, 0, 0, 1, 0] == pytest.approx(3)  # sigmoid(-0.01) < 0.5
    assert above.Q[0, 0, 0, 1, 0] > 3  # sigmoid(0.4) > 0.5


# --------------------------------------------------------------------- initial bias


def _biased_pso(pso_config, decoder=None, swarm=400):
    pso_config["solver"]["pso"].update(
        {
            "lot_sizing_bounds": {"enabled": False},
            "execution_backend": "python",
            "jit_warmup": False,
            "swarm_size": swarm,
            "max_iterations": 0,
        }
    )
    if decoder is not None:
        pso_config["solver"]["pso"]["decoder"] = decoder
    return ParticleSwarmOptimization(
        _data(customers=3, products=2, vehicles=2, periods=3), "/tmp", Mock()
    )


def test_initial_bias_shifts_only_the_genes_with_a_threshold(pso_config):
    pso = _biased_pso(pso_config)
    positions = pso._random_positions()
    heuristic = pso.heuristic
    delivery = positions[:, heuristic.delivery_gene_indices]
    production = positions[:, : heuristic.dim_x]
    unused = np.delete(
        positions[:, heuristic.dim_x :],
        heuristic.delivery_gene_indices - heuristic.dim_x,
        axis=1,
    )
    assert abs(delivery.mean() + 3.0) < 0.25
    assert abs(production.mean() + 1.0) < 0.25
    assert abs(unused.mean()) < 0.1
    assert positions.dtype == np.float32


def test_initial_bias_is_not_applied_to_the_legacy_decoder(pso_config):
    legacy = {
        "setup_threshold": 0.0,
        "routing": "nearest_neighbor",
        "delivery": "just_in_time",
    }
    positions = _biased_pso(pso_config, legacy)._random_positions()
    assert abs(positions.mean()) < 0.1


def test_initial_bias_can_be_disabled_and_tuned(pso_config):
    pso_config["solver"]["pso"]["initial_bias"] = {"delivery": 0.0, "production": 2.0}
    pso = _biased_pso(pso_config)
    positions = pso._random_positions()
    heuristic = pso.heuristic
    assert abs(positions[:, heuristic.delivery_gene_indices].mean()) < 0.25
    assert abs(positions[:, : heuristic.dim_x].mean() + 2.0) < 0.25


def test_resampled_and_reinitialized_particles_keep_the_bias(pso_config):
    pso = _biased_pso(pso_config, swarm=200)
    pso._initialize_swarm()
    heuristic = pso.heuristic
    pso._stagnation_count = 10**6
    pso.pso_config["reinitialize_fraction"] = 1.0
    pso.pso_config["elite_fraction"] = 0.0
    pso._reinitialize_if_stagnant()
    delivery = pso.positions[:, heuristic.delivery_gene_indices]
    assert abs(delivery.mean() + 3.0) < 0.3


def test_pso_defaults_for_the_initial_bias(pso_config):
    assert _biased_pso(pso_config).pso_config["initial_bias"] == {
        "delivery": 3.0,
        "production": 1.0,
    }
    assert Config.normalize({})["solver"]["pso"]["initial_bias"] == {
        "delivery": 3.0,
        "production": 1.0,
    }


@pytest.mark.parametrize(
    "bias",
    [{"delivery": -1}, {"production": True}, {"production": "1"}, {"speed": 1.0}],
)
def test_config_rejects_invalid_initial_bias(bias):
    with pytest.raises(ValueError, match="initial_bias"):
        Config.normalize({"solver": {"pso": {"initial_bias": bias}}})
