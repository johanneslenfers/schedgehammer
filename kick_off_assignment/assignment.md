# Kick-Off Assigment 

Write a simple Auto-Tuning tool that supports different parameters types and uses some optimization method beyond Random Sampling. 

## Parameter Types
Use the so-called RIOC + Permutation parameter types.
  - Switch: [True, False] -> True
  - Real: [0, 1] -> 0.33
  - Integer: [1, 1024] -> 7
  - Ordinal: [1, 2, 4, 8, 16, 32] -> 32
  - Categorical: ["A", "B", "C"] -> "C"
  - Permutation: [1, 2, 3, 4, 5] -> [1, 3, 2, 4, 5]

## Optimization Method
Leverage an optimization method that is more sophisticated than Random Sampling.  

## Example Function 
Cost function stub that takes a configuration and returns a performance value. 
```python
def cost(configuration: Dict[str, Parameter]) -> float:

  # find example in repo (cost.py)
  
  return 0.0
```

## Development 
Create a branch in the project-seminar repository. Once you’ve finished the task, submit a pull request.