"""Heuristic separation of cumulative rounded-capacity inequalities.

The separator intentionally has no dependency on DOcplex or CPLEX.  It receives
the fractional route variables from the exact model and returns candidate cuts
that can be materialized by the caller's callback.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class RoundedCapacityCut:
    """A cumulative rounded-capacity cut identified by prefix and customers."""

    tau: int
    customers: tuple[int, ...]
    rhs: int
    violation: float

    @property
    def key(self):
        return self.tau, self.customers


def build_obligatory_demands(demand, initial_inventory):
    """Return obligatory cumulative delivery by prefix and customer.

    ``demand`` has parser layout ``[product][customer][period]`` and
    ``initial_inventory`` has layout ``[product][node]`` with node 0 as the
    plant.  The returned array has layout ``[prefix][customer]``.

    Product inventories are kept separate before aggregation so that surplus
    stock of one product cannot incorrectly offset demand of another product.
    """

    demand = np.asarray(demand, dtype=float)
    initial_inventory = np.asarray(initial_inventory, dtype=float)
    if demand.ndim != 3:
        raise ValueError("demand deve ter formato [produto, cliente, periodo]")
    if initial_inventory.ndim != 2:
        raise ValueError("initial_inventory deve ter formato [produto, no]")
    products, customers, periods = demand.shape
    if initial_inventory.shape[0] != products:
        raise ValueError("demanda e estoque inicial devem ter o mesmo numero de produtos")
    if initial_inventory.shape[1] < customers + 1:
        raise ValueError("estoque inicial deve conter planta e todos os clientes")

    cumulative = np.cumsum(demand, axis=2)
    initial_customers = initial_inventory[:, 1 : customers + 1, None]
    residual = np.maximum(cumulative - initial_customers, 0.0)
    return residual.sum(axis=0).T


def aggregate_route_arcs(z_values, tau):
    """Aggregate directed route values into an undirected prefix matrix.

    ``z_values`` has layout ``[period][vehicle][node][node]``.  The returned
    matrix contains both directions of each pair exactly once when it is used
    through :func:`cut_value`.
    """

    z_values = np.asarray(z_values, dtype=float)
    if z_values.ndim != 4:
        raise ValueError("z_values deve ter formato [periodo, veiculo, no, no]")
    if tau < 0 or tau >= z_values.shape[0]:
        raise ValueError("tau fora do horizonte de periodos")

    directed = z_values[: tau + 1].sum(axis=(0, 1))
    aggregated = directed + directed.T
    np.fill_diagonal(aggregated, 0.0)
    return aggregated


def cut_value(aggregated_arcs, customers, customer_node_offset=0):
    """Return the fractional number of arcs crossing a customer subset."""

    selected = np.zeros(aggregated_arcs.shape[0], dtype=bool)
    selected[[customer + customer_node_offset for customer in customers]] = True
    return float(aggregated_arcs[selected][:, ~selected].sum())


def _rounded_rhs(obligatory_volume, capacity, tolerance):
    if capacity <= 0:
        raise ValueError("capacity deve ser positiva")
    if obligatory_volume <= tolerance:
        return 0
    return 2 * int(math.ceil(obligatory_volume / capacity - tolerance))


def _cut_candidate(
    aggregated_arcs,
    obligatory,
    tau,
    customers,
    capacity,
    tolerance,
    customer_node_offset,
    statistics=None,
):
    if statistics is not None:
        statistics["candidate_evaluations"] = statistics.get("candidate_evaluations", 0) + 1
    customers = tuple(sorted(customers))
    volume = float(obligatory[tau, list(customers)].sum())
    rhs = _rounded_rhs(volume, capacity, tolerance)
    if rhs == 0:
        return None
    violation = float(
        rhs - cut_value(aggregated_arcs, customers, customer_node_offset)
    )
    return RoundedCapacityCut(tau, customers, rhs, violation)


def _candidate_moves(aggregated_arcs, customers, all_customers):
    selected = set(customers)
    outside = all_customers - selected
    for customer in outside:
        yield tuple(sorted((*selected, customer)))
    for customer in tuple(selected):
        yield tuple(sorted(selected - {customer}))
    for removed in tuple(selected):
        for added in outside:
            yield tuple(sorted((selected - {removed}) | {added}))


def _rounded_rhs_array(volumes, capacity, tolerance):
    if capacity <= 0:
        raise ValueError("capacity deve ser positiva")
    volumes = np.asarray(volumes, dtype=float)
    rhs = np.zeros(volumes.shape, dtype=int)
    positive = volumes > tolerance
    rhs[positive] = 2 * np.ceil(
        volumes[positive] / capacity - tolerance
    ).astype(int)
    return rhs


def _improve_subset(
    aggregated_arcs,
    obligatory,
    tau,
    seed,
    capacity,
    tolerance,
    customer_node_offset,
    max_moves,
    statistics=None,
):
    """Grow and locally improve one subset with vectorized move scoring.

    The crossing value of a subset is computed from row sums and internal
    customer arcs.  This avoids materializing a tuple and a boolean cut mask
    for every add/remove/swap candidate.
    """

    if statistics is not None:
        statistics["candidate_evaluations"] = statistics.get(
            "candidate_evaluations", 0
        ) + 1

    customer_count = obligatory.shape[1]
    customer_nodes = np.arange(customer_count) + customer_node_offset
    customer_arcs = aggregated_arcs[np.ix_(customer_nodes, customer_nodes)]
    row_sums = aggregated_arcs[customer_nodes].sum(axis=1)
    demand_at_tau = obligatory[tau]

    current = np.array(sorted(seed), dtype=int)
    selected_mask = np.zeros(customer_count, dtype=bool)
    selected_mask[current] = True
    current_volume = float(demand_at_tau[current].sum())
    current_lhs = float(
        row_sums[current].sum()
        - customer_arcs[np.ix_(current, current)].sum()
    )
    current_rhs = int(
        _rounded_rhs(current_volume, capacity, tolerance)
    )
    current_result = (
        RoundedCapacityCut(
            tau,
            tuple(current.tolist()),
            current_rhs,
            float(current_rhs - current_lhs),
        )
        if current_rhs
        else None
    )
    best = current_result

    for _ in range(max_moves):
        outside = np.flatnonzero(~selected_mask)
        candidates = []

        if outside.size:
            add_lhs = current_lhs + row_sums[outside]
            if current.size:
                add_lhs -= 2.0 * customer_arcs[np.ix_(outside, current)].sum(axis=1)
            add_volumes = current_volume + demand_at_tau[outside]
            add_rhs = _rounded_rhs_array(add_volumes, capacity, tolerance)
            add_violations = add_rhs - add_lhs
            if statistics is not None:
                statistics["candidate_evaluations"] = statistics.get(
                    "candidate_evaluations", 0
                ) + int(outside.size)
            add_index = int(np.argmax(add_violations))
            candidates.append(
                (
                    float(add_violations[add_index]),
                    int(add_rhs[add_index]),
                    tuple(sorted((*current.tolist(), int(outside[add_index])))),
                    float(add_volumes[add_index]),
                    float(add_lhs[add_index]),
                    "add",
                    -1,
                    int(outside[add_index]),
                )
            )

        if current.size > 1:
            remove_lhs = current_lhs - row_sums[current]
            for position, customer in enumerate(current):
                remaining = np.delete(current, position)
                if remaining.size:
                    remove_lhs[position] += 2.0 * customer_arcs[
                        customer, remaining
                    ].sum()
            remove_volumes = current_volume - demand_at_tau[current]
            remove_rhs = _rounded_rhs_array(remove_volumes, capacity, tolerance)
            remove_violations = remove_rhs - remove_lhs
            if statistics is not None:
                statistics["candidate_evaluations"] = statistics.get(
                    "candidate_evaluations", 0
                ) + int(current.size)
            remove_index = int(np.argmax(remove_violations))
            remove_customer = int(current[remove_index])
            candidates.append(
                (
                    float(remove_violations[remove_index]),
                    int(remove_rhs[remove_index]),
                    tuple(
                        int(customer)
                        for customer in current
                        if int(customer) != remove_customer
                    ),
                    float(remove_volumes[remove_index]),
                    float(remove_lhs[remove_index]),
                    "remove",
                    remove_customer,
                    -1,
                )
            )

        if current.size and outside.size:
            best_swap = None
            for removed_position, removed in enumerate(current):
                remaining = np.delete(current, removed_position)
                swap_lhs = current_lhs - row_sums[removed]
                if remaining.size:
                    swap_lhs += 2.0 * customer_arcs[removed, remaining].sum()
                swap_lhs = (
                    swap_lhs
                    + row_sums[outside]
                    - 2.0 * customer_arcs[np.ix_(outside, remaining)].sum(axis=1)
                )
                swap_volumes = (
                    current_volume
                    - demand_at_tau[removed]
                    + demand_at_tau[outside]
                )
                swap_rhs = _rounded_rhs_array(swap_volumes, capacity, tolerance)
                swap_violations = swap_rhs - swap_lhs
                if statistics is not None:
                    statistics["candidate_evaluations"] = statistics.get(
                        "candidate_evaluations", 0
                    ) + int(outside.size)
                swap_index = int(np.argmax(swap_violations))
                candidate = (
                    float(swap_violations[swap_index]),
                    int(swap_rhs[swap_index]),
                    tuple(
                        sorted(
                            (*remaining.tolist(), int(outside[swap_index]))
                        )
                    ),
                    float(swap_volumes[swap_index]),
                    float(swap_lhs[swap_index]),
                    "swap",
                    int(removed),
                    int(outside[swap_index]),
                )
                if best_swap is None or candidate[:3] > best_swap[:3]:
                    best_swap = candidate
            if best_swap is not None:
                candidates.append(best_swap)

        if not candidates:
            break
        next_candidate = max(candidates, key=lambda candidate: candidate[:3])
        current_violation = (
            current_result.violation if current_result is not None else -float("inf")
        )
        if next_candidate[0] > current_violation + tolerance:
            selected = list(next_candidate[2])
        else:
            # Growth may be temporarily non-improving while crossing a
            # rounding plateau, so choose the largest-volume add move.
            growth = [candidate for candidate in candidates if candidate[5] == "add"]
            if not growth:
                break
            next_candidate = max(growth, key=lambda candidate: (candidate[3], candidate[0]))
            selected = list(next_candidate[2])

        current = np.array(selected, dtype=int)
        selected_mask[:] = False
        selected_mask[current] = True
        current_volume = next_candidate[3]
        current_lhs = next_candidate[4]
        current_rhs = next_candidate[1]
        current_result = (
            RoundedCapacityCut(
                tau,
                tuple(current.tolist()),
                current_rhs,
                next_candidate[0],
            )
            if current_rhs
            else None
        )
        if current_result is not None and (
            best is None or current_result.violation > best.violation + tolerance
        ):
            best = current_result

    return best


def _build_seeds(aggregated_arcs, obligatory, tau, customer_node_offset):
    customers = set(range(obligatory.shape[1]))
    positive = [
        customer
        for customer in customers
        if obligatory[tau, customer] > 0
    ]
    seeds = {tuple([customer]) for customer in positive}

    pair_scores = []
    for left in positive:
        for right in positive:
            if left >= right:
                continue
            internal = float(
                aggregated_arcs[
                    left + customer_node_offset, right + customer_node_offset
                ]
            )
            pair_scores.append((internal, left, right))
    for _, left, right in sorted(pair_scores, reverse=True)[: max(10, len(positive))]:
        seeds.add((left, right))
    return sorted(seeds)


def separate_cumulative_cuts(
    z_values,
    obligatory_demands,
    capacity,
    max_cuts=10,
    min_violation=1e-6,
    tolerance=1e-9,
    customer_node_offset=0,
    statistics=None,
):
    """Find the best heuristic cumulative cuts for a fractional route point."""

    z_values = np.asarray(z_values, dtype=float)
    obligatory_demands = np.asarray(obligatory_demands, dtype=float)
    if obligatory_demands.ndim != 2:
        raise ValueError("obligatory_demands deve ter formato [prefixo, cliente]")
    if z_values.shape[0] != obligatory_demands.shape[0]:
        raise ValueError("Z e demandas devem ter o mesmo numero de periodos")
    if customer_node_offset < 0 or customer_node_offset + obligatory_demands.shape[1] > z_values.shape[2]:
        raise ValueError("Z deve conter todos os nos dos clientes")
    if max_cuts <= 0:
        return []

    if statistics is not None:
        statistics["prefixes"] = int(z_values.shape[0])
        statistics["seeds"] = 0
        statistics["candidate_evaluations"] = 0
    candidates = {}
    for tau in range(z_values.shape[0]):
        aggregated = aggregate_route_arcs(z_values, tau)
        seeds = _build_seeds(
            aggregated, obligatory_demands, tau, customer_node_offset
        )
        if statistics is not None:
            statistics["seeds"] = statistics.get("seeds", 0) + len(seeds)
        for seed in seeds:
            result = _improve_subset(
                aggregated,
                obligatory_demands,
                tau,
                seed,
                capacity,
                tolerance,
                customer_node_offset,
                # Keep each callback inexpensive: seeds provide diversification,
                # while two moves are enough to grow or exchange a small subset.
                max_moves=min(2, max(1, obligatory_demands.shape[1])),
                statistics=statistics,
            )
            if result is None or result.violation <= min_violation:
                continue
            previous = candidates.get(result.key)
            if previous is None or result.violation > previous.violation:
                candidates[result.key] = result

    return sorted(
        candidates.values(),
        key=lambda result: (-result.violation, result.tau, result.customers),
    )[:max_cuts]
