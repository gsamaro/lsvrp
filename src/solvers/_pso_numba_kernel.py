import math

import numpy as np
from numba import njit, prange


@njit(cache=True, inline="always")
def _sigmoid(value):
    if math.isnan(value):
        return 0.5
    bounded = min(60.0, max(-60.0, value))
    return 1.0 / (1.0 + math.exp(-bounded))


@njit(cache=True, inline="always")
def _next_random(state):
    state ^= state >> np.uint64(12)
    state ^= state << np.uint64(25)
    state ^= state >> np.uint64(27)
    value = state * np.uint64(2685821657736338717)
    return state, float(value >> np.uint64(11)) * (1.0 / 9007199254740992.0)


@njit(cache=True)
def _route_distance(nodes, length, distance):
    total = 0.0
    previous = 0
    for index in range(length):
        total += distance[previous, nodes[index]]
        previous = nodes[index]
    return total + distance[previous, 0]


@njit(cache=True)
def _improve_route(nodes, length, distance):
    """2-opt and single-node relocation (Or-opt) until no move shortens the route."""
    improved = True
    while improved:
        improved = False
        for first in range(length - 1):
            for last in range(first + 1, length):
                before = distance[0 if first == 0 else nodes[first - 1], nodes[first]]
                after = distance[0 if first == 0 else nodes[first - 1], nodes[last]]
                before += distance[
                    nodes[last], 0 if last == length - 1 else nodes[last + 1]
                ]
                after += distance[
                    nodes[first], 0 if last == length - 1 else nodes[last + 1]
                ]
                if after < before - 1e-9:
                    low = first
                    high = last
                    while low < high:
                        swap = nodes[low]
                        nodes[low] = nodes[high]
                        nodes[high] = swap
                        low += 1
                        high -= 1
                    improved = True
        for source in range(length):
            node = nodes[source]
            previous = 0 if source == 0 else nodes[source - 1]
            following = 0 if source == length - 1 else nodes[source + 1]
            removal = (
                distance[previous, node]
                + distance[node, following]
                - distance[previous, following]
            )
            best_gain = 1e-9
            best_slot = -1
            for slot in range(length + 1):
                if slot == source or slot == source + 1:
                    continue
                left = 0 if slot == 0 else nodes[slot - 1]
                right = 0 if slot == length else nodes[slot]
                insertion = (
                    distance[left, node] + distance[node, right] - distance[left, right]
                )
                if removal - insertion > best_gain:
                    best_gain = removal - insertion
                    best_slot = slot
            if best_slot >= 0:
                if best_slot > source:
                    for index in range(source, best_slot - 1):
                        nodes[index] = nodes[index + 1]
                    nodes[best_slot - 1] = node
                else:
                    for index in range(source, best_slot, -1):
                        nodes[index] = nodes[index - 1]
                    nodes[best_slot] = node
                improved = True
                break


@njit(cache=True)
def _savings_partition(
    customers, count, load, distance, capacity, vehicle_count, assignments, remaining
):
    """Clarke-Wright partition of ``customers`` (ascending) into capacity routes.

    Fills ``assignments``/``remaining`` and returns True when at most
    ``vehicle_count`` routes result; otherwise leaves them untouched.
    """
    routes = np.zeros((count, count), dtype=np.int64)
    lengths = np.zeros(count, dtype=np.int64)
    route_load = np.zeros(count, dtype=np.float64)
    route_of = np.full(distance.shape[0], -1, dtype=np.int64)
    for index in range(count):
        customer = customers[index]
        routes[index, 0] = customer
        lengths[index] = 1
        route_load[index] = load[customer]
        route_of[customer] = index

    pairs = count * (count - 1) // 2
    saving = np.zeros(pairs, dtype=np.float64)
    first_of = np.zeros(pairs, dtype=np.int64)
    second_of = np.zeros(pairs, dtype=np.int64)
    slot = 0
    for left in range(count):
        for right in range(left + 1, count):
            i = customers[left]
            j = customers[right]
            saving[slot] = -(distance[0, i] + distance[j, 0] - distance[i, j])
            first_of[slot] = i
            second_of[slot] = j
            slot += 1
    for position in np.argsort(saving[:pairs], kind="mergesort"):
        i = first_of[position]
        j = second_of[position]
        a = route_of[i]
        b = route_of[j]
        if a == b or route_load[a] + route_load[b] > capacity + 1e-8:
            continue
        a_end = lengths[a] - 1
        b_end = lengths[b] - 1
        i_first = routes[a, 0] == i
        i_last = routes[a, a_end] == i
        j_first = routes[b, 0] == j
        j_last = routes[b, b_end] == j
        if not (i_first or i_last) or not (j_first or j_last):
            continue
        if not i_last:
            for index in range(lengths[a] // 2):
                swap = routes[a, index]
                routes[a, index] = routes[a, a_end - index]
                routes[a, a_end - index] = swap
        if not j_first:
            for index in range(lengths[b] // 2):
                swap = routes[b, index]
                routes[b, index] = routes[b, b_end - index]
                routes[b, b_end - index] = swap
        for index in range(lengths[b]):
            routes[a, lengths[a] + index] = routes[b, index]
            route_of[routes[b, index]] = a
        lengths[a] += lengths[b]
        route_load[a] += route_load[b]
        lengths[b] = 0
        route_load[b] = 0.0

    used = 0
    for index in range(count):
        if lengths[index] > 0:
            used += 1
    if used > vehicle_count:
        return False
    # Vehicle labels follow the smallest customer of each route (deterministic).
    taken = np.zeros(count, dtype=np.uint8)
    for vehicle in range(used):
        chosen = -1
        chosen_min = 0
        for index in range(count):
            if lengths[index] == 0 or taken[index]:
                continue
            smallest = routes[index, 0]
            for member in range(1, lengths[index]):
                if routes[index, member] < smallest:
                    smallest = routes[index, member]
            if chosen < 0 or smallest < chosen_min:
                chosen = index
                chosen_min = smallest
        taken[chosen] = 1
        for member in range(lengths[chosen]):
            assignments[routes[chosen, member]] = vehicle
    # Same subtraction order as the Python reference: ascending customer index.
    for index in range(count):
        customer = customers[index]
        remaining[assignments[customer]] -= load[customer]
    return True


@njit(cache=True)
def _plan_ahead(
    row,
    dim_x,
    product_count,
    vehicle_count,
    location_count,
    period_count,
    period,
    previous_inventory,
    demand,
    cumulative,
    inventory_capacity,
    deficits,
    ahead,
    fleet_cap,
    vehicle_capacity,
    guard_active,
    threshold,
    max_cover,
):
    """Fill ``ahead[p, c]`` with deliveries beyond the period deficit.

    Two sources, both clipped by customer storage, vehicle capacity and the
    usable fleet capacity ``fleet_cap``, so they never make a particle invalid:

    * load levelling (``guard_active``): the amount that must be shipped now so
      no later period needs more than ``fleet_cap``, taken from the nearest
      deadlines first and spread over the customers in proportion to their need;
    * cover genes: a customer whose gene is above ``threshold`` receives enough
      to cover the demand of the next ``k <= max_cover`` periods.
    """
    stock = np.zeros((product_count, location_count), dtype=np.float64)
    mandatory = 0.0
    for product in range(product_count):
        for customer in range(1, location_count):
            value = (
                previous_inventory[product, customer]
                + deficits[product, customer]
                - demand[product, customer - 1, period]
            )
            stock[product, customer] = value if value > 0 else 0.0
            mandatory += deficits[product, customer]

    if guard_active and period < period_count - 1:
        need = np.zeros(period_count - period, dtype=np.float64)
        for later in range(period + 1, period_count):
            total = 0.0
            for product in range(product_count):
                for customer in range(1, location_count):
                    value = (
                        cumulative[product, customer, later]
                        - cumulative[product, customer, period]
                        - stock[product, customer]
                    )
                    if value > 0:
                        total += value
            need[later - period] = total
        carry = 0.0
        for later in range(period_count - 1, period, -1):
            load = need[later - period] - need[later - period - 1] + carry
            carry = load - fleet_cap if load > fleet_cap else 0.0
        spare = fleet_cap - mandatory
        if spare < 0:
            spare = 0.0
        remaining = carry if carry < spare else spare
        for later in range(period + 1, period_count):
            if remaining <= 1e-9:
                break
            wanted = np.zeros((product_count, location_count), dtype=np.float64)
            wanted_total = 0.0
            for customer in range(1, location_count):
                for product in range(product_count):
                    base = stock[product, customer] + ahead[product, customer]
                    covered = (
                        cumulative[product, customer, later]
                        - cumulative[product, customer, period]
                    )
                    before = (
                        cumulative[product, customer, later - 1]
                        - cumulative[product, customer, period]
                    )
                    value = (covered - base if covered > base else 0.0) - (
                        before - base if before > base else 0.0
                    )
                    room = inventory_capacity[product, customer] - base
                    if room < value:
                        value = room
                    if value > 0:
                        wanted[product, customer] = value
                        wanted_total += value
            if wanted_total <= 1e-9:
                continue
            # Spread the shipment over every customer with demand at this deadline,
            # which keeps the loads even and the routes easier to pack.
            factor = 1.0 if wanted_total <= remaining else remaining / wanted_total
            for customer in range(1, location_count):
                for product in range(product_count):
                    amount = wanted[product, customer] * factor
                    if amount > 0:
                        ahead[product, customer] += amount
                        remaining -= amount

    loaded = mandatory
    for product in range(product_count):
        for customer in range(1, location_count):
            loaded += ahead[product, customer]
    for customer in range(1, location_count):
        gene = row[
            dim_x
            + (
                ((0 * vehicle_count + 0) * location_count + customer) * period_count
                + period
            )
        ]
        share = _sigmoid(gene)
        if share <= threshold:
            continue
        cover = 1 + int(math.floor((share - threshold) / (1.0 - threshold) * max_cover))
        if cover > max_cover:
            cover = max_cover
        last = period + cover
        if last > period_count - 1:
            last = period_count - 1
        if last <= period:
            continue
        customer_total = 0.0
        for product in range(product_count):
            customer_total += deficits[product, customer] + ahead[product, customer]
        for product in range(product_count):
            target = (
                cumulative[product, customer, last]
                - cumulative[product, customer, period]
            )
            if target > inventory_capacity[product, customer]:
                target = inventory_capacity[product, customer]
            add = target - (stock[product, customer] + ahead[product, customer])
            if add <= 0:
                continue
            room_customer = vehicle_capacity - customer_total
            if add > room_customer:
                add = room_customer
            room_fleet = fleet_cap - loaded
            if add > room_fleet:
                add = room_fleet
            if add <= 1e-6:
                continue
            ahead[product, customer] += add
            customer_total += add
            loaded += add


@njit(cache=True, parallel=True)
def update_swarm_kernel(
    positions,
    velocities,
    personal_best_positions,
    global_best_position,
    random_states,
    inertia,
    cognitive,
    social,
    velocity_limit,
):
    for particle in prange(positions.shape[0]):
        state = random_states[particle]
        for gene in range(positions.shape[1]):
            state, r1 = _next_random(state)
            state, r2 = _next_random(state)
            position = positions[particle, gene]
            velocity = (
                inertia * velocities[particle, gene]
                + cognitive * r1 * (personal_best_positions[particle, gene] - position)
                + social * r2 * (global_best_position[gene] - position)
            )
            if velocity > velocity_limit:
                velocity = velocity_limit
            elif velocity < -velocity_limit:
                velocity = -velocity_limit
            velocities[particle, gene] = velocity
            positions[particle, gene] = position + velocity
        random_states[particle] = state


@njit(cache=True, parallel=True)
def build_population_kernel(
    positions,
    product_count,
    vehicle_count,
    location_count,
    period_count,
    production_capacity,
    vehicle_capacity,
    big_m,
    production_time,
    production_cost,
    setup_cost,
    inventory_cost,
    inventory_capacity,
    initial_inventory,
    demand,
    distance,
    fixed_vehicle_cost,
    lower_x,
    upper_x,
    strengthened_production_upper,
    lower_q,
    upper_q,
    has_x_bounds,
    has_q_bounds,
    setup_threshold,
    routing_mode,
    delivery_mode,
    delivery_threshold,
    max_cover,
    fleet_cap,
    guard_active,
    cumulative_demand,
):
    population_size = positions.shape[0]
    customer_count = location_count - 1
    dim_x = product_count * period_count

    x_all = np.zeros((population_size, product_count, period_count), dtype=np.float64)
    y_all = np.zeros((population_size, product_count, period_count), dtype=np.int8)
    i_all = np.zeros(
        (population_size, product_count, location_count, period_count), dtype=np.float64
    )
    q_all = np.zeros(
        (
            population_size,
            product_count,
            vehicle_count,
            location_count,
            period_count,
        ),
        dtype=np.float64,
    )
    assignments_all = np.full(
        (population_size, period_count, location_count), -1, dtype=np.int16
    )
    route_nodes_all = np.zeros(
        (population_size, period_count, vehicle_count, customer_count), dtype=np.int16
    )
    route_lengths_all = np.zeros(
        (population_size, period_count, vehicle_count), dtype=np.int16
    )
    feasible_all = np.ones(population_size, dtype=np.uint8)
    costs = np.full(population_size, np.inf, dtype=np.float64)
    x_fingerprints = np.zeros(population_size, dtype=np.uint64)
    q_quantity_fingerprints = np.zeros(population_size, dtype=np.uint64)
    assignment_fingerprints = np.zeros(population_size, dtype=np.uint64)
    q_full_fingerprints = np.zeros(population_size, dtype=np.uint64)

    for particle in prange(population_size):
        previous_inventory = initial_inventory.copy()
        feasible = True

        for period in range(period_count):
            deficits = np.zeros((product_count, location_count), dtype=np.float64)
            production = np.zeros(product_count, dtype=np.float64)
            deliveries = np.zeros((product_count, location_count), dtype=np.float64)

            for product in range(product_count):
                for customer in range(1, location_count):
                    value = (
                        demand[product, customer - 1, period]
                        - previous_inventory[product, customer]
                    )
                    if value < 0:
                        value = 0
                    deficits[product, customer] = value
                    deliveries[product, customer] = value

            if delivery_mode == 1:
                ahead = np.zeros((product_count, location_count), dtype=np.float64)
                _plan_ahead(
                    positions[particle],
                    dim_x,
                    product_count,
                    vehicle_count,
                    location_count,
                    period_count,
                    period,
                    previous_inventory,
                    demand,
                    cumulative_demand,
                    inventory_capacity,
                    deficits,
                    ahead,
                    fleet_cap,
                    vehicle_capacity,
                    guard_active,
                    delivery_threshold,
                    max_cover,
                )
                # Deliveries ahead are produced to order, so clip them to the
                # production time left after the mandatory deliveries.
                mandatory_time = 0.0
                total_time = 0.0
                for product in range(product_count):
                    mandatory_total = 0
                    ahead_total = 0
                    for customer in range(1, location_count):
                        mandatory_total += deficits[product, customer]
                        ahead_total += (
                            deficits[product, customer] + ahead[product, customer]
                        )
                    need_mandatory = mandatory_total - previous_inventory[product, 0]
                    if need_mandatory < 0:
                        need_mandatory = 0
                    need_total = ahead_total - previous_inventory[product, 0]
                    if need_total < 0:
                        need_total = 0
                    mandatory_time += production_time[product] * need_mandatory
                    total_time += production_time[product] * need_total
                scale = 1.0
                if (
                    total_time > production_capacity + 1e-8
                    and total_time > mandatory_time
                ):
                    scale = (production_capacity - mandatory_time) / (
                        total_time - mandatory_time
                    )
                    if scale < 0:
                        scale = 0.0
                for product in range(product_count):
                    for customer in range(1, location_count):
                        deliveries[product, customer] = (
                            deficits[product, customer]
                            + scale * ahead[product, customer]
                        )

            for product in range(product_count):
                total_deficit = 0
                for customer in range(1, location_count):
                    total_deficit += deliveries[product, customer]
                required = total_deficit - previous_inventory[product, 0]
                if required < 0:
                    required = 0
                production[product] = required
                if has_x_bounds:
                    bound = lower_x[product, period]
                    if bound > production[product]:
                        production[product] = bound
                if (
                    production[product]
                    > strengthened_production_upper[product, period] + 1e-8
                ):
                    feasible = False
                    break

            if not feasible:
                break

            used_time = 0
            for product in range(product_count):
                used_time += production_time[product] * production[product]
            if used_time > production_capacity + 1e-8:
                feasible = False
                break

            remaining_time = production_capacity - used_time
            product_order = np.arange(product_count)
            for left in range(product_count - 1):
                best = left
                for right in range(left + 1, product_count):
                    candidate = product_order[right]
                    current = product_order[best]
                    candidate_gene = _sigmoid(
                        positions[particle, candidate * period_count + period]
                    )
                    current_gene = _sigmoid(
                        positions[particle, current * period_count + period]
                    )
                    if candidate_gene > current_gene or (
                        candidate_gene == current_gene and candidate < current
                    ):
                        best = right
                if best != left:
                    temp = product_order[left]
                    product_order[left] = product_order[best]
                    product_order[best] = temp

            for order_index in range(product_count):
                product = product_order[order_index]
                if production_time[product] <= 0:
                    continue
                max_extra = remaining_time / production_time[product]
                strengthened_extra = float(
                    strengthened_production_upper[product, period] - production[product]
                )
                if strengthened_extra < max_extra:
                    max_extra = strengthened_extra
                if max_extra < -1e-8:
                    feasible = False
                    break
                gene = positions[particle, product * period_count + period]
                share = _sigmoid(gene)
                if setup_threshold > 0.0:
                    # Below the threshold the particle produces only what is needed,
                    # so a period can end up without a setup.
                    if share <= setup_threshold:
                        share = 0.0
                    else:
                        share = (share - setup_threshold) / (1.0 - setup_threshold)
                if has_x_bounds:
                    decoded = float(
                        lower_x[product, period]
                        + share * (upper_x[product, period] - lower_x[product, period])
                    )
                    extra = decoded - production[product]
                    if extra < 0:
                        extra = 0
                    if extra > max_extra:
                        extra = max_extra
                else:
                    storage_room = (
                        inventory_capacity[product, 0] - previous_inventory[product, 0]
                    )
                    if storage_room < 0:
                        storage_room = 0
                    high = min(storage_room, max_extra)
                    extra = float(share * high)
                if setup_threshold > 0.0 and extra < 1e-6:
                    extra = 0.0
                production[product] += extra
                remaining_time -= production_time[product] * extra

            for product in range(product_count):
                if production[product] > big_m:
                    feasible = False
                    break
            if not feasible:
                break

            customer_order = np.zeros(customer_count, dtype=np.int64)
            customer_load = np.zeros(location_count, dtype=np.float64)
            active_count = 0
            for customer in range(1, location_count):
                total = 0
                for product in range(product_count):
                    total += deliveries[product, customer]
                customer_load[customer] = total
                if total > vehicle_capacity + 1e-8:
                    feasible = False
                    break
                if total > 0:
                    customer_order[active_count] = customer
                    active_count += 1
            if not feasible:
                break

            for left in range(active_count - 1):
                best = left
                for right in range(left + 1, active_count):
                    candidate = customer_order[right]
                    current = customer_order[best]
                    if customer_load[candidate] > customer_load[current] or (
                        customer_load[candidate] == customer_load[current]
                        and candidate < current
                    ):
                        best = right
                if best != left:
                    temp = customer_order[left]
                    customer_order[left] = customer_order[best]
                    customer_order[best] = temp

            assignments = np.full(location_count, -1, dtype=np.int16)
            remaining_vehicle = np.full(
                vehicle_count, vehicle_capacity, dtype=np.float64
            )
            partitioned = False
            if routing_mode == 1 and active_count > 0:
                ascending = np.zeros(active_count, dtype=np.int64)
                filled = 0
                for customer in range(1, location_count):
                    if customer_load[customer] > 0:
                        ascending[filled] = customer
                        filled += 1
                partitioned = _savings_partition(
                    ascending,
                    active_count,
                    customer_load,
                    distance,
                    vehicle_capacity,
                    vehicle_count,
                    assignments,
                    remaining_vehicle,
                )
            for customer_index in range(0 if partitioned else active_count):
                customer = customer_order[customer_index]
                total = customer_load[customer]
                selected_vehicle = -1
                selected_remainder = 0
                selected_preference = 0.0
                for vehicle in range(vehicle_count):
                    if remaining_vehicle[vehicle] + 1e-8 < total:
                        continue
                    remainder = remaining_vehicle[vehicle] - total
                    preference = 0.0
                    for product in range(product_count):
                        q_index = dim_x + (
                            (
                                (product * vehicle_count + vehicle) * location_count
                                + customer
                            )
                            * period_count
                            + period
                        )
                        preference += positions[particle, q_index]
                    preference = _sigmoid(preference)
                    if (
                        selected_vehicle < 0
                        or remainder < selected_remainder
                        or (
                            remainder == selected_remainder
                            and (
                                preference > selected_preference
                                or (
                                    preference == selected_preference
                                    and vehicle < selected_vehicle
                                )
                            )
                        )
                    ):
                        selected_vehicle = vehicle
                        selected_remainder = remainder
                        selected_preference = preference
                if selected_vehicle < 0:
                    feasible = False
                    break
                assignments[customer] = selected_vehicle
                remaining_vehicle[selected_vehicle] -= total
            if not feasible:
                break

            excess = np.zeros(product_count, dtype=np.float64)
            headroom_total = np.zeros(product_count, dtype=np.float64)
            for product in range(product_count):
                mandatory = 0
                room_total = 0
                for customer in range(1, location_count):
                    mandatory += deliveries[product, customer]
                    customer_inventory = (
                        previous_inventory[product, customer]
                        + deliveries[product, customer]
                        - demand[product, customer - 1, period]
                    )
                    room = inventory_capacity[product, customer] - customer_inventory
                    if room > 0:
                        room_total += room
                remaining_at_plant = (
                    previous_inventory[product, 0] + production[product] - mandatory
                )
                if remaining_at_plant < -1e-8:
                    feasible = False
                    break
                excess[product] = remaining_at_plant - inventory_capacity[product, 0]
                if excess[product] < 0:
                    excess[product] = 0
                headroom_total[product] = room_total
            if not feasible:
                break

            drain_order = np.arange(product_count)
            for left in range(product_count - 1):
                best = left
                for right in range(left + 1, product_count):
                    candidate = drain_order[right]
                    current = drain_order[best]
                    if headroom_total[candidate] < headroom_total[current] or (
                        headroom_total[candidate] == headroom_total[current]
                        and candidate < current
                    ):
                        best = right
                if best != left:
                    temp = drain_order[left]
                    drain_order[left] = drain_order[best]
                    drain_order[best] = temp

            for order_index in range(product_count):
                product = drain_order[order_index]
                while excess[product] > 1e-8:
                    selected_customer = -1
                    selected_vehicle = -1
                    selected_available = 0
                    selected_preference = -1e300
                    selected_negative_capacity = -9223372036854775807
                    for customer in range(1, location_count):
                        customer_inventory = (
                            previous_inventory[product, customer]
                            + deliveries[product, customer]
                            - demand[product, customer - 1, period]
                        )
                        stock_room = (
                            inventory_capacity[product, customer] - customer_inventory
                        )
                        if stock_room <= 0:
                            continue
                        first_vehicle = assignments[customer]
                        vehicle_start = 0 if first_vehicle < 0 else first_vehicle
                        vehicle_end = (
                            vehicle_count if first_vehicle < 0 else first_vehicle + 1
                        )
                        for vehicle in range(vehicle_start, vehicle_end):
                            available = min(stock_room, remaining_vehicle[vehicle])
                            if available <= 0:
                                continue
                            q_index = dim_x + (
                                (
                                    (product * vehicle_count + vehicle) * location_count
                                    + customer
                                )
                                * period_count
                                + period
                            )
                            preference = _sigmoid(positions[particle, q_index])
                            if has_q_bounds:
                                desired_value = 0.0
                                for desired_vehicle in range(vehicle_count):
                                    desired_index = dim_x + (
                                        (
                                            (product * vehicle_count + desired_vehicle)
                                            * location_count
                                            + customer
                                        )
                                        * period_count
                                        + period
                                    )
                                    desired_value += lower_q[
                                        product, desired_vehicle, customer, period
                                    ] + _sigmoid(positions[particle, desired_index]) * (
                                        upper_q[
                                            product, desired_vehicle, customer, period
                                        ]
                                        - lower_q[
                                            product, desired_vehicle, customer, period
                                        ]
                                    )
                                if desired_value > deliveries[product, customer]:
                                    preference += 1.0
                            negative_capacity = -remaining_vehicle[vehicle]
                            if (
                                selected_customer < 0
                                or preference > selected_preference
                                or (
                                    preference == selected_preference
                                    and (
                                        negative_capacity > selected_negative_capacity
                                        or (
                                            negative_capacity
                                            == selected_negative_capacity
                                            and (
                                                customer > selected_customer
                                                or (
                                                    customer == selected_customer
                                                    and (
                                                        vehicle > selected_vehicle
                                                        or (
                                                            vehicle == selected_vehicle
                                                            and available
                                                            > selected_available
                                                        )
                                                    )
                                                )
                                            )
                                        )
                                    )
                                )
                            ):
                                selected_customer = customer
                                selected_vehicle = vehicle
                                selected_available = available
                                selected_preference = preference
                                selected_negative_capacity = negative_capacity
                    if selected_customer < 0:
                        feasible = False
                        break
                    quantity = min(excess[product], selected_available)
                    deliveries[product, selected_customer] += quantity
                    remaining_vehicle[selected_vehicle] -= quantity
                    assignments[selected_customer] = selected_vehicle
                    excess[product] -= quantity
                if not feasible:
                    break
            if not feasible:
                break

            for product in range(product_count):
                x_all[particle, product, period] = production[product]
                y_all[particle, product, period] = 1 if production[product] > 0 else 0
            for customer in range(1, location_count):
                vehicle = assignments[customer]
                assignments_all[particle, period, customer] = vehicle
                if vehicle >= 0:
                    for product in range(product_count):
                        q_all[particle, product, vehicle, customer, period] = (
                            deliveries[product, customer]
                        )

            for product in range(product_count):
                delivered = 0
                for customer in range(1, location_count):
                    delivered += deliveries[product, customer]
                plant_inventory = (
                    previous_inventory[product, 0] + production[product] - delivered
                )
                i_all[particle, product, 0, period] = plant_inventory
                if (
                    plant_inventory < -1e-8
                    or plant_inventory > inventory_capacity[product, 0] + 1e-8
                ):
                    feasible = False
                    break
                for customer in range(1, location_count):
                    inventory = (
                        previous_inventory[product, customer]
                        + deliveries[product, customer]
                        - demand[product, customer - 1, period]
                    )
                    i_all[particle, product, customer, period] = inventory
                    if (
                        inventory < -1e-8
                        or inventory > inventory_capacity[product, customer] + 1e-8
                    ):
                        feasible = False
                        break
                if not feasible:
                    break
            if not feasible:
                break

            for vehicle in range(vehicle_count):
                visited = np.zeros(location_count, dtype=np.uint8)
                current = 0
                route_length = 0
                while True:
                    selected_customer = -1
                    selected_distance = 0
                    selected_q_preference = 0.0
                    for customer in range(1, location_count):
                        if assignments[customer] != vehicle or visited[customer]:
                            continue
                        q_preference = 0.0
                        for product in range(product_count):
                            q_index = dim_x + (
                                (
                                    (product * vehicle_count + vehicle) * location_count
                                    + customer
                                )
                                * period_count
                                + period
                            )
                            q_preference += positions[particle, q_index]
                        candidate_distance = distance[current, customer]
                        if (
                            selected_customer < 0
                            or candidate_distance < selected_distance
                            or (
                                candidate_distance == selected_distance
                                and (
                                    q_preference > selected_q_preference
                                    or (
                                        q_preference == selected_q_preference
                                        and customer < selected_customer
                                    )
                                )
                            )
                        ):
                            selected_customer = customer
                            selected_distance = candidate_distance
                            selected_q_preference = q_preference
                    if selected_customer < 0:
                        break
                    route_nodes_all[particle, period, vehicle, route_length] = (
                        selected_customer
                    )
                    route_length += 1
                    visited[selected_customer] = 1
                    current = selected_customer
                if routing_mode == 1 and route_length > 1:
                    _improve_route(
                        route_nodes_all[particle, period, vehicle],
                        route_length,
                        distance,
                    )
                route_lengths_all[particle, period, vehicle] = route_length

            previous_inventory = i_all[particle, :, :, period].copy()

        if feasible:
            total_cost = 0.0
            for product in range(product_count):
                for period in range(period_count):
                    total_cost += setup_cost[product] * y_all[particle, product, period]
                    total_cost += (
                        production_cost[product] * x_all[particle, product, period]
                    )
                    for location in range(location_count):
                        total_cost += (
                            inventory_cost[product, location]
                            * i_all[particle, product, location, period]
                        )
            for period in range(period_count):
                for vehicle in range(vehicle_count):
                    length = route_lengths_all[particle, period, vehicle]
                    if length <= 0:
                        continue
                    total_cost += fixed_vehicle_cost
                    previous = 0
                    for route_index in range(length):
                        customer = route_nodes_all[
                            particle, period, vehicle, route_index
                        ]
                        total_cost += distance[previous, customer]
                        previous = customer
                    total_cost += distance[previous, 0]
            costs[particle] = total_cost
            x_hash = np.uint64(1469598103934665603)
            quantity_hash = np.uint64(1469598103934665603)
            assignment_hash = np.uint64(1469598103934665603)
            q_hash = np.uint64(1469598103934665603)
            prime = np.uint64(1099511628211)
            for period in range(period_count):
                for product in range(product_count):
                    x_hash = (
                        x_hash
                        ^ np.uint64(round(x_all[particle, product, period] * 1e6))
                    ) * prime
                    for customer in range(1, location_count):
                        quantity = 0
                        for vehicle in range(vehicle_count):
                            value = q_all[particle, product, vehicle, customer, period]
                            quantity += value
                            q_hash = (q_hash ^ np.uint64(round(value * 1e6))) * prime
                        quantity_hash = (
                            quantity_hash ^ np.uint64(round(quantity * 1e6))
                        ) * prime
                for customer in range(1, location_count):
                    assignment_hash = (
                        assignment_hash
                        ^ np.uint64(assignments_all[particle, period, customer] + 1)
                    ) * prime
            x_fingerprints[particle] = x_hash
            q_quantity_fingerprints[particle] = quantity_hash
            assignment_fingerprints[particle] = assignment_hash
            q_full_fingerprints[particle] = q_hash
        else:
            feasible_all[particle] = 0
            x_all[particle].fill(0)
            y_all[particle].fill(0)
            i_all[particle].fill(0)
            q_all[particle].fill(0)
            assignments_all[particle].fill(-1)
            route_nodes_all[particle].fill(0)
            route_lengths_all[particle].fill(0)

    return (
        x_all,
        y_all,
        i_all,
        q_all,
        assignments_all,
        route_nodes_all,
        route_lengths_all,
        feasible_all,
        costs,
        x_fingerprints,
        q_quantity_fingerprints,
        assignment_fingerprints,
        q_full_fingerprints,
    )


@njit(cache=True, parallel=True)
def population_components(
    x, y, inventory, nodes, lengths, c, s, h, distance, fixed_cost
):
    components = np.zeros((x.shape[0], 5, x.shape[2]), dtype=np.float64)
    for row in prange(x.shape[0]):
        for t in range(x.shape[2]):
            for p in range(x.shape[1]):
                components[row, 0, t] += c[p] * x[row, p, t]
                components[row, 1, t] += s[p] * y[row, p, t]
                for i in range(inventory.shape[2]):
                    components[row, 2, t] += h[p, i] * inventory[row, p, i, t]
            for v in range(lengths.shape[2]):
                length = lengths[row, t, v]
                if length:
                    components[row, 3, t] += fixed_cost
                    previous = 0
                    for k in range(length):
                        customer = nodes[row, t, v, k]
                        components[row, 4, t] += distance[previous, customer]
                        previous = customer
                    components[row, 4, t] += distance[previous, 0]
    return components
