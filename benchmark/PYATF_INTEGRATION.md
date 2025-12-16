# pyATF Integration with Catbench

This document explains how to use pyATF with catbench benchmarks.

## Overview

The `pyatf_from_study()` function in `interop_problem.py` translates catbench `Study` objects into pyATF tuning parameters and cost functions.

## Parameter Translation

| Catbench Type | pyATF Representation | Notes |
|---------------|---------------------|-------|
| `Integer` | `TP(name, Interval(min, max))` | Direct mapping |
| `Real` | `TP(name, Interval(min, max))` | Direct mapping |
| `Boolean` | `TP(name, Interval(0, 1))` | 0=False, 1=True |
| `Categorical` | `TP(name, Set(*categories))` | Uses pyATF's Set with all category values |
| `IntExponential` | `TP(name, Interval(min_exp, max_exp))` | Stores exponents, transformed in cost function |
| `Permutation` | `TP(name, Set(*permutations))` | Uses pyATF's Set with all possible permutations |

## Constraints

**Important**: Constraint translation from catbench's string-based format to pyATF's lambda functions is **not yet automated**.

Catbench constraints are expressions like:
```python
"block_size <= num_threads"
```

pyATF constraints are lambda functions in the `TP` definition:
```python
TP('block_size', Interval(1, 1024), 
   lambda block_size, num_threads: block_size <= num_threads)
```

### Manual Constraint Translation

If your benchmark has constraints, you'll need to manually add them to the tuning parameters:

1. Check the warning printed by `pyatf_from_study()`
2. Identify which parameters are involved in each constraint
3. Add constraint lambda to the relevant `TP` definition

Example:
```python
tuning_params, cost_function = pyatf_from_study(study)

# Manually update constraints if needed
# Find the TP for 'block_size' and add constraint
for tp in tuning_params:
    if tp.name == 'block_size':
        # Add interdependent constraint
        tp.constraint = lambda block_size, num_threads: block_size <= num_threads
```

## Usage Example

```python
import catbench as cb
from interop_problem import pyatf_from_study
from pyatf import Tuner, Evaluations
from pyatf.search_techniques import AUCBandit

# Load benchmark
study = cb.benchmark("mttkrp")

# Convert to pyATF
tuning_params, cost_function = pyatf_from_study(study)

# Run tuning
result = Tuner().tuning_parameters(*tuning_params) \
                .search_technique(AUCBandit()) \
                .tune(cost_function, Evaluations(100))

print(f"Best cost: {result.costs[result.best_configuration]}")
print(f"Best config: {result.best_configuration}")
```

## Running the Benchmark

```bash
python benchmark/pyatf_benchmark.py
```

## Search Techniques Available

- `AUCBandit()` - Recommended default (meta-technique)
- `Random()` - Random search
- `Exhaustive()` - Exhaustive search
- `SimulatedAnnealing()` - Simulated annealing
- `DifferentialEvolution()` - Differential evolution
- `PatternSearch()` - Pattern search
- `Torczon()` - Torczon's simplex method

## Abort Conditions

- `Evaluations(n)` - Stop after n evaluations
- `Duration(timedelta(hours=1))` - Stop after time limit
- `Fraction(0.1)` - Stop after exploring 10% of search space
- `Cost(threshold)` - Stop when cost <= threshold
- `Or(...)` / `And(...)` - Combine conditions

## Cost Function

The cost function automatically:
1. Transforms exponential parameters from log space
2. Converts list-based permutations to string format if needed by catbench
3. Queries catbench with the configuration (categorical and permutation values are used directly)
4. Returns the `compute_time` metric
5. Raises `CostFunctionError` if query fails (penalizes invalid configs)

## Limitations

1. **Permutation parameters** generate all possible permutations (factorial growth) - may be very large for long sequences
2. **Constraint translation** is manual - string expressions must be converted to lambdas
3. **Fidelity parameters** use default values from catbench study
4. **Multi-objective tuning** (if needed) requires custom cost function

## Future Improvements

- Automatic constraint parsing and translation
- Smart permutation handling for large search spaces (e.g., sampling or encoding)
- Multi-objective cost functions
- Parallel evaluation support
