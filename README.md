# Schedgehammer: Auto-Tuning Compiler Optimizations Beyond Numerical Parameters

## Goal of the Research Project

Schedgehammer is a general-purpose auto-scheduling framework designed to optimize program schedules across diverse compiler infrastructures. Unlike existing auto-schedulers that are tightly coupled to specific intermediate representations or rely on template-based search, Schedgehammer introduces a **generic and reusable representation of optimization schedules** that can be integrated with multiple user-schedulable languages (USLs).

The key research goal is to **decouple auto-scheduling from specific compiler infrastructures**, treating scheduling as a generic optimization problem. This allows different USLs (such as TVM and TACO) to benefit from a shared and extensible tuning infrastructure without requiring compiler-specific assumptions or extensive reengineering.

### Key Contributions

- A generic abstraction of compiler scheduling as an optimization problem, independent of specific compiler representations
- An implementation and interface layer that integrates with existing USLs
- Empirical evaluation showing substantial performance improvements over baseline schedules across multiple benchmarks
- Analysis of schedule structure, identifying key factors that influence performance in diverse scheduling spaces

## Methodology

Schedgehammer treats scheduling as an optimization problem defined by an abstract search space and a cost function, iteratively searching for improved configurations in an optimization loop. The system supports two main optimization strategies:

### 1. Random Search
A baseline optimization method that explores the search space by sampling configurations uniformly at random. For numerical parameters, Schedgehammer leverages a constraint solver to generate valid configurations. For scheduling parameters, random tuning proceeds by iteratively selecting and applying random scheduling operations whose function signatures are compatible with the current schedule state.

### 2. Genetic Tuning
An evolutionary optimization approach that maintains a population of candidate solutions, evaluates their performance, and generates new candidates through selection, crossover, and mutation. The tuning budget is split into two phases:
- **Exploration phase**: Random schedules are generated and evaluated to build a diverse population
- **Exploitation phase**: The most effective schedules are selected as the elite and further mutated (refining primitive parameters like tile sizes or split factors)

### Constraint Handling

Schedgehammer employs a constraint solver that enforces both individual parameter bounds and inter-parameter relationships. Constraints are parsed through an ANTLR-generated parser, allowing the constraint language to be easily expanded. The solver uses depth-first search to suggest valid configurations, filtering out invalid values until a fixpoint is reached.

### Parameter Types

The framework supports multiple tuning parameter types:
- **Real Number**: Continuous values with user-defined step-width for discretization
- **Integer**: Discrete integer values
- **Permutation**: Permutations of sequences
- **Ordinal**: Ordered categorical values (e.g., optimization levels "O0", "O1", "O2", "O3")
- **Categorical**: Unordered categorical values
- **Exponential Integer**: Powers of a base (e.g., 2^x for x in {2..8})

## Representation of Schedules

Schedgehammer represents schedules as **graph-structured objects** that allow systematic mutation, validation, and composition of transformations. The representation consists of two key components:

### 1. Schedule Parameter

A schedule parameter models the operations of a USL (User-Schedulable Language). It consists of:
- A `create_schedule` function that creates the initial schedule and returns the schedule object and initial axes
- A `finish_schedule` function that takes the schedule object and returns the compiled schedule ready for execution
- Minimum and maximum count of applied schedule operations
- Available schedule operations (e.g., TILE, REORDER, SPLIT, FUSE)

### 2. Graph Representation

Internally, Schedgehammer represents an optimization schedule as a **bipartite graph** containing two types of nodes:

1. **Operation nodes** (square): Correspond to scheduling operations such as `tile`, `reorder`, or `split`
2. **Data nodes** (round): Represent operation arguments and return values (loop axes or scalar parameters)

Edges capture dependencies between operations and their inputs or outputs, forming a directed dataflow structure. A topological ordering is maintained through incremental annotations that record the relative execution order of operations.


### Example: Matrix Multiplication Schedule

For a matrix multiplication in TVM, a schedule might apply tiling and reordering:

```python
i, j = C.op.axis
io, jo, ii, ji = s[C].tile(i, j, tile_m, tile_n)
s[C].reorder(io, jo, k, ii, ji)
```

This corresponds to a graph where:
- Input axes `i` and `j` serve as arguments to the `tile` operation
- The `tile` operation returns four new axes (io, jo, ii, ji)
- The subsequent `reorder` operation takes these axes along with the reduction axis `k` as inputs

### Dynamic Search Space

Schedule auto-tuning differs fundamentally from classical parameter tuning because it operates on a **dynamic, stateful search space** rather than a static parameter space. Each applied operation mutates the schedule and modifies the set of valid operations that can follow. Schedgehammer handles this dependency by leveraging the graph representation: when an operation is applied, the available axes and their connections are updated accordingly.

## Results

Schedgehammer was evaluated through two case studies:

### Case Study 1: Parameter Tuning

Schedgehammer was evaluated on traditional parameter auto-tuning tasks using the Catbench benchmarking suite, comparing against OpenTuner and PyATF on three benchmarks:

- **Harris Corner Detection**: Image processing benchmark optimizing tile-sizes, vector-widths, and global/local sizes
- **MTTKRP** (Matricized Tensor Times Khatri-Rao Product): Core operation in tensor decomposition
- **SpMV** (Sparse Matrix-Vector Multiplication): Widely used kernel in sparse linear algebra

**Results**: Across all benchmarks, Schedgehammer's methods showed robustness, benefiting from its ability to handle both constraints and permutation parameters. While competitors could converge faster when they succeeded, they often failed in certain scenarios (e.g., OpenTuner failed on Harris due to lack of constraint support) where Schedgehammer remained reliable. This indicates that Schedgehammer delivers consistent, near-optimal performance.

![Figure 4: Results of Schedgehammer numeric tuning on benchmark tests](diagrams/figure_page_7.png)

### Case Study 2: Auto-Scheduling

Schedgehammer was evaluated on auto-scheduling tasks using TVM and TACO frameworks:

#### TVM Benchmarks

- **Matrix Multiplication**: Schedgehammer tuners quickly plateaued at ~0.8 seconds (from 1.22s baseline), while Ansor continued improving to ~0.3 seconds. This can be explained by Ansor having more optimization types available and being specifically developed for this benchmark.

- **MTTKRP**: Schedgehammer tuners and Ansor performed very similarly, improving execution time from 0.2 seconds baseline to about 0.035 seconds.

- **2D Convolution**: **Schedgehammer clearly outperformed Ansor**, surpassing Ansor's final result in less than 10 iterations. Ansor took about 35 iterations to beat the baseline and ended around 0.23 seconds.

#### TACO Benchmarks

- **GEMM**: Improved baseline from 3.4 seconds to 3.3 seconds
- **MTTKRP**: Improved baseline from 3.15 seconds to about 3.0 seconds
- **SpMV**: Minimal improvement over the 0.013 seconds baseline

![Figure 5: Results of Schedgehammer schedule tuning on benchmark tests with Taco and TVM](diagrams/figure_page_8.png)

### Key Findings

1. **Genetic vs. Random**: Genetic tuning was not significantly more effective than random generation for schedule optimization. Analysis revealed that while slow schedules cannot be turned into fast schedules by tuning primitive parameters only, top-performing schedules with different primitive parameters are quickly turned into average-performing ones. This explains why genetic tuning (which picks good schedules and tunes their primitive parameters) does not work as well as anticipated.

![Figure 6(a): Effects of mutating primitive parameters on schedule runtime for Matrix Multiplication (TVM)](diagrams/page_10_img_0.png)

The figure above shows the results of an experiment where 500 different random schedules for matrix multiplication were created, and for each schedule, 50 variants were generated by randomly changing primitive parameters. The schedules are sorted by performance from best (left) to worst (right). The key findings are: (1) slow schedules cannot be turned into fast schedules by tuning primitive parameters only, and (2) top-performing schedules (index 0-50) with different primitive parameters are quickly turned into average-performing ones. This explains why genetic tuning does not work as well as anticipated: picking a good schedule (index 50-150) and tuning its primitive parameters is more likely to produce another good or average schedule than producing a top-performing one.

2. **Schedule Structure Matters**: The structure of the schedule itself (the sequence and types of operations) has a more significant impact on performance than the primitive parameters (like tile sizes). This suggests that future work should focus on better search strategies for schedule structure rather than parameter refinement.

3. **Flexibility and Generality**: Schedgehammer successfully optimizes both dense and sparse computations in constrained and unconstrained search spaces, without relying on compiler-specific assumptions, demonstrating its flexibility and generality.

## Future Work

The paper identifies several directions for future research:

1. **Schedgehammer as a Research Playground**: The graph-based representation provides a rich foundation for analyzing how different operations and their orderings influence performance across different hardware architectures.

2. **Beating Random Generation**: More sophisticated search strategies are needed, such as:
   - Learned cost models for performance prediction
   - Transfer learning from previous tuning sessions
   - Hybrid strategies combining random generation with lightweight heuristics

3. **Schedule Embeddings**: Exploring common subgraphs or motifs among high-performing schedules could enable the development of learned heuristics or schedule embeddings.

