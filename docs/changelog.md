# Changelog

## 2026-10-04

### New

- Added a configurable deterministic MPPRP matheuristic with a shared TSP order, restricted routing, route improvement, and optional exact or PSO initialization.
- Added complementary stage outputs, an offline debug notebook, and the mathematical and implementation documentation.
- Added unit coverage for fractional quantities, routing stages, solver handoffs, and debug outputs.

### Improvements

- Standardized production, inventory, delivery, and load quantities as continuous `float64` values across the exact solver and PSO.
- Added optional PSO iteration time limit and kept matheuristic stage budgets independent.

### Fixes

- Preserved valid best solutions when a later exact solve has no better feasible incumbent.
- Kept stage-specific gaps from being reported as bounds for the unrestricted MPPRP.
