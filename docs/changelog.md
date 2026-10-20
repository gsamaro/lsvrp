# Changelog

## 2026-10-20

### New

- Added `experiments` to `config/config.json`: a list of `{name, overrides}` run one after another in a single job (a single queue submission on the cluster). Each experiment is the base configuration with its overrides deep-merged on top, runs to completion including its `union_results`, and writes under `<instance.output>/<name>/`. Overrides of `instance.output`, `workers` and `experiments` are rejected, and every experiment is validated before anything runs. Without `experiments` the behavior is unchanged.
- `src/process/ExperimentRunner.py` keeps an `experiment_state.json` per experiment and resumes safely: it skips an experiment only after opening the exact `union_results` recorded in the state and checking the experiment name, the configuration fingerprint and the number of tasks. A changed configuration moves the old folder to `<name>.stale-<timestamp>` (nothing is deleted). A failing experiment does not stop the next ones and the job exits with code 1.
- The consolidated `union_results` has pairing keys: `experiment`, `experiment_fingerprint`, `experiment_run_tag`, `instance_key`, `task_key` (`instance|weight|alpha`) and `row_key` (`task_key|period`). `task_key` and `row_key` are the same in every experiment for the same instance, weight vector and alpha.
- MPI workers apply the resolved configuration carried by each task (`config_json`), so persistent ranks follow the experiment being run, and one `MPIPoolExecutor` is shared by all experiments. `WorkerProcess.run_parallel` accepts an `executor`, and `Config.experiments()` / `Config.replace()` were added.
- Added `solver.total_time_budget` (default `false`): `timeLimit` becomes the total budget of a task and every CPLEX call receives only the time left (`src/helpers/TimeBudget.py`).
- Added `solver.polishing.after_seconds` (default `null`): starts CPLEX solution polishing on the unrestricted MIP after that many seconds (counted from the task start with a time budget).
- `config/config.json` now holds the production setup for the 20-customer instances (1 thread, 3600 s per task, vehicle-free model, `objective_scale` 1e5, single-threaded PSO) and three experiments: `pso_mip_polish`, `matheuristic_mip_polish` and `matheuristic`.

### Fixes

- The matheuristic PSO start now uses the thread limit of the run instead of every available core.

### Notes

- The tests of the vehicle-indexed model no longer depend on `config/config.json`.

## 2026-10-19

### New

- Added `solver.objective_scale` (default `1.0`; `config/config.json` uses `100000.0`): the multiobjective goal programming objective is multiplied by this factor inside CPLEX, and every objective value and bound read back (`FO`, `objBound`, `relaxedModelObjVal`, bound and incumbent telemetry, matheuristic stage records) is divided by it. `FO`, bound, and gap keep their previous unit.

### Fixes

- With `alpha = 0.99` the marginal cost of a decision variable (about `(1 - alpha) * w * c / target`, between 1e-10 and 1e-5 for the coefficients of `p`) is below the CPLEX optimality tolerance, so the dual and primal simplex stopped before the LP optimum. On the aggregated model the LP value was 3.9% (`PRP13_C20_P8_V5_T12_S4`), 9.7% (`PRP10_C20_P8_V5_T12_S2`) and 7.5% (`PRP19_C30_P10_V5_T12_S4`) above the exact value given by barrier, so the reported lower bound was above the valid one. With a factor of 1e4 the three algorithms agree within 3e-6 (relative), and from 1e5 on they match in every digit measured.

### Notes

- A change of variable `q = p / target` was tried and dropped: it keeps the objective coefficients between 0.002 and 0.99, but the reduced cost of the decision variables is unchanged and the dual simplex stayed inexact.

## 2026-10-10

### New

- Added `solver.aggregate_vehicles` (default `false`): the exact model can drop the vehicle index from `Z`, `R`, and `Q`, bounding plant departures by the fleet size, and rebuilds one route per vehicle on output. The output tuple, Excel, and Parquet contracts are unchanged.
- Added `expand_aggregated_vehicles` and `aggregate_vehicle_axis` to split fleet-wide routes into vehicles and to sum a vehicle-indexed MIP start.
- Added unit and CPLEX integration coverage that compares the aggregated and vehicle-indexed optima.
- Added `decoder.delivery = lookahead` and `solver.pso.initial_bias`: deliveries ahead of the deficit through load levelling and per-customer cover genes, clipped by customer storage, vehicle capacity, fleet capacity, and production time. No instance of the 801 in the repository is left without a feasible particle (103 before); the delivery genes now change the cost and the delivery profile diversity goes from 1 to hundreds.
- Added `solver.pso.decoder` (`setup_threshold`, `routing`): production genes can now skip a setup, customers are partitioned into routes by Clarke-Wright savings, and every route is improved with 2-opt and Or-opt. The numba kernel and the Python reference agree on every option. The legacy decoder remains available.
- Added `solver.matheuristic.start` (`legacy`, `decoder`, `pso`; default `decoder`). The restricted stage used to start from a particle of zero genes with the legacy decoder (objective 24.6 on `PRP13_C20_P8_V5_T12_S4`), and its 137k-variable vehicle-indexed MIP never improved it.
- With `solver.aggregate_vehicles`, the matheuristic stages use the vehicle-free model: the restricted stage keeps binary arcs along the TSP order (`arc_order`, 29k variables instead of 137k) and the route improvement fixes the plan on all arcs (`fixed_plan` without `eta`). The restricted optimum matches the vehicle-indexed one on small instances.
- The route improvement now minimizes the configured objective under goal programming: a lower transport total could still be a worse solution because `lambda` is a maximum.
- Numerical residue in the delivery totals (near 1e-12) no longer counts as a delivery: it made the improvement start infeasible and allowed zero-delivery cycles.

### Improvements

- Matheuristic alone on `PRP13_C20_P8_V5_T12_S4`: 24.6 to 0.25. Across four 20-customer instances it matched or beat PSO followed by the full MIP in three of four at a similar or lower total time (mean 0.2375 against 0.2630, one run each, noisy under CPU contention); the full MIP after it did not improve the solution in 100 seconds.
- Full MIP started from the PSO solution on `PRP13_C20_P8_V5_T12_S4` (5 minutes, 2 threads): the reported gap fell from 100% (vehicle-indexed model, root relaxation did not finish) to 70% with `aggregate_vehicles` and the old PSO, and to 30% with `aggregate_vehicles` and the new PSO (objective 0.2405, bound 0.1685).
- PSO cost on `PRP13_C20_P8_V5_T12_S4` dropped from 1.20 to 0.35; across 12 instances of 20 customers the mean cost went from 0.73 to 0.28.
- `config/config.json` disables `lot_sizing_bounds`: with them on, the PSO only returned the LP vertex (`X = lower`), and they rescued no infeasible instance.

### Notes

- On `PRP13_C20_P8_V5_T12_S4` the delivery lookahead does not lower the PSO cost (five seeds: 0.327 before, 0.335 after); its value is feasibility on tight instances and diversity. Only one of the delivery genes per customer and period is read.
- The matheuristic stages, relaxed target construction, and HC1/VC do not use the aggregated model; Coelho inequalities and rounded capacity cuts remain valid with it.
- Documented that the objective coefficients of `p_t^j` are below the default CPLEX optimality tolerance, so an unscaled LP can return a value above its optimum.

## 2026-10-04

### New

- Added a configurable deterministic MPPRP matheuristic with a shared TSP order, restricted routing, route improvement, and optional exact or PSO initialization.
- Added complementary stage outputs, an offline debug notebook, and the mathematical and implementation documentation.
- Added unit coverage for fractional quantities, routing stages, solver handoffs, and debug outputs.

### Improvements

- Standardized production, inventory, delivery, and load quantities as continuous `float64` values across the exact solver and PSO.
- Added optional PSO iteration time limit and kept matheuristic stage budgets independent.
- Expanded the debug notebook with run discovery and period-by-period demand service and inventory balance analysis.

### Fixes

- Preserved valid best solutions when a later exact solve has no better feasible incumbent.
- Kept stage-specific gaps from being reported as bounds for the unrestricted MPPRP.
