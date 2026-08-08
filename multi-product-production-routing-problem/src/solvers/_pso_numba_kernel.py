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
def _round_int(value):
    return int(round(value))


@njit(cache=True, inline="always")
def _next_random(state):
    state ^= state >> np.uint64(12)
    state ^= state << np.uint64(25)
    state ^= state >> np.uint64(27)
    value = state * np.uint64(2685821657736338717)
    return state, float(value >> np.uint64(11)) * (1.0 / 9007199254740992.0)


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
                + cognitive
                * r1
                * (personal_best_positions[particle, gene] - position)
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
    lower_q,
    upper_q,
    has_x_bounds,
    has_q_bounds,
):
    population_size = positions.shape[0]
    customer_count = location_count - 1
    dim_x = product_count * period_count

    x_all = np.zeros((population_size, product_count, period_count), dtype=np.int64)
    y_all = np.zeros((population_size, product_count, period_count), dtype=np.int64)
    i_all = np.zeros(
        (population_size, product_count, location_count, period_count), dtype=np.int64
    )
    q_all = np.zeros(
        (
            population_size,
            product_count,
            vehicle_count,
            location_count,
            period_count,
        ),
        dtype=np.int64,
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
            deficits = np.zeros((product_count, location_count), dtype=np.int64)
            production = np.zeros(product_count, dtype=np.int64)
            deliveries = np.zeros((product_count, location_count), dtype=np.int64)

            for product in range(product_count):
                total_deficit = 0
                for customer in range(1, location_count):
                    value = demand[product, customer - 1, period] - previous_inventory[product, customer]
                    if value < 0:
                        value = 0
                    deficits[product, customer] = value
                    deliveries[product, customer] = value
                    total_deficit += value
                required = total_deficit - previous_inventory[product, 0]
                if required < 0:
                    required = 0
                production[product] = required
                if has_x_bounds:
                    bound = int(math.ceil(lower_x[product, period]))
                    if bound > production[product]:
                        production[product] = bound

            used_time = 0
            for product in range(product_count):
                used_time += production_time[product] * production[product]
            if used_time > production_capacity:
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
                max_extra = remaining_time // production_time[product]
                gene = positions[particle, product * period_count + period]
                if has_x_bounds:
                    decoded = _round_int(
                        lower_x[product, period]
                        + _sigmoid(gene)
                        * (upper_x[product, period] - lower_x[product, period])
                    )
                    extra = decoded - production[product]
                    if extra < 0:
                        extra = 0
                    if extra > max_extra:
                        extra = max_extra
                else:
                    storage_room = inventory_capacity[product, 0] - previous_inventory[product, 0]
                    if storage_room < 0:
                        storage_room = 0
                    high = min(storage_room, max_extra)
                    extra = _round_int(_sigmoid(gene) * high)
                production[product] += extra
                remaining_time -= production_time[product] * extra

            for product in range(product_count):
                if production[product] > big_m:
                    feasible = False
                    break
            if not feasible:
                break

            customer_order = np.zeros(customer_count, dtype=np.int64)
            customer_load = np.zeros(location_count, dtype=np.int64)
            active_count = 0
            for customer in range(1, location_count):
                total = 0
                for product in range(product_count):
                    total += deliveries[product, customer]
                customer_load[customer] = total
                if total > vehicle_capacity:
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
            remaining_vehicle = np.full(vehicle_count, vehicle_capacity, dtype=np.int64)
            for customer_index in range(active_count):
                customer = customer_order[customer_index]
                total = customer_load[customer]
                selected_vehicle = -1
                selected_remainder = 0
                selected_preference = 0.0
                for vehicle in range(vehicle_count):
                    if remaining_vehicle[vehicle] < total:
                        continue
                    remainder = remaining_vehicle[vehicle] - total
                    preference = 0.0
                    for product in range(product_count):
                        q_index = dim_x + (
                            ((product * vehicle_count + vehicle) * location_count + customer)
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

            excess = np.zeros(product_count, dtype=np.int64)
            headroom_total = np.zeros(product_count, dtype=np.int64)
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
                remaining_at_plant = previous_inventory[product, 0] + production[product] - mandatory
                if remaining_at_plant < 0:
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
                while excess[product] > 0:
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
                        stock_room = inventory_capacity[product, customer] - customer_inventory
                        if stock_room <= 0:
                            continue
                        first_vehicle = assignments[customer]
                        vehicle_start = 0 if first_vehicle < 0 else first_vehicle
                        vehicle_end = vehicle_count if first_vehicle < 0 else first_vehicle + 1
                        for vehicle in range(vehicle_start, vehicle_end):
                            available = min(stock_room, remaining_vehicle[vehicle])
                            if available <= 0:
                                continue
                            q_index = dim_x + (
                                ((product * vehicle_count + vehicle) * location_count + customer)
                                * period_count
                                + period
                            )
                            preference = _sigmoid(positions[particle, q_index])
                            if has_q_bounds:
                                desired_value = 0.0
                                for desired_vehicle in range(vehicle_count):
                                    desired_index = dim_x + (
                                        ((product * vehicle_count + desired_vehicle) * location_count + customer)
                                        * period_count
                                        + period
                                    )
                                    desired_value += lower_q[
                                        product, desired_vehicle, customer, period
                                    ] + _sigmoid(positions[particle, desired_index]) * (
                                        upper_q[product, desired_vehicle, customer, period]
                                        - lower_q[product, desired_vehicle, customer, period]
                                    )
                                if _round_int(desired_value) > deliveries[product, customer]:
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
                                            negative_capacity == selected_negative_capacity
                                            and (
                                                customer > selected_customer
                                                or (
                                                    customer == selected_customer
                                                    and (
                                                        vehicle > selected_vehicle
                                                        or (
                                                            vehicle == selected_vehicle
                                                            and available > selected_available
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
                        q_all[particle, product, vehicle, customer, period] = deliveries[
                            product, customer
                        ]

            for product in range(product_count):
                delivered = 0
                for customer in range(1, location_count):
                    delivered += deliveries[product, customer]
                plant_inventory = previous_inventory[product, 0] + production[product] - delivered
                i_all[particle, product, 0, period] = plant_inventory
                if plant_inventory < 0 or plant_inventory > inventory_capacity[product, 0]:
                    feasible = False
                    break
                for customer in range(1, location_count):
                    inventory = (
                        previous_inventory[product, customer]
                        + deliveries[product, customer]
                        - demand[product, customer - 1, period]
                    )
                    i_all[particle, product, customer, period] = inventory
                    if inventory < 0 or inventory > inventory_capacity[product, customer]:
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
                                ((product * vehicle_count + vehicle) * location_count + customer)
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
                    route_nodes_all[
                        particle, period, vehicle, route_length
                    ] = selected_customer
                    route_length += 1
                    visited[selected_customer] = 1
                    current = selected_customer
                route_lengths_all[particle, period, vehicle] = route_length

            previous_inventory = i_all[particle, :, :, period].copy()

        if feasible:
            total_cost = 0.0
            for product in range(product_count):
                for period in range(period_count):
                    total_cost += setup_cost[product] * y_all[particle, product, period]
                    total_cost += production_cost[product] * x_all[particle, product, period]
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
                        customer = route_nodes_all[particle, period, vehicle, route_index]
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
                    x_hash = (x_hash ^ np.uint64(x_all[particle, product, period])) * prime
                    for customer in range(1, location_count):
                        quantity = 0
                        for vehicle in range(vehicle_count):
                            value = q_all[particle, product, vehicle, customer, period]
                            quantity += value
                            q_hash = (q_hash ^ np.uint64(value)) * prime
                        quantity_hash = (quantity_hash ^ np.uint64(quantity)) * prime
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
