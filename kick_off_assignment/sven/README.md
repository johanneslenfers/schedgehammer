## Usecase
    Someone has written a program and now specifies the parameters he wants to tune
    by id and type  and value range


## How to run 
```python tune.py <parameter definition>.yml <time in seconds> <exponent i for shifting the comma in the time budget>```

e.g. ```python tune.py params.yaml 1 -5``` for a time budget of 1*10^-5 seconds

## Grouping
It's possible to specify groups of params in the yaml to tune the groups independently of each other

## General approach
For each group
1. Calculate amount of possible value combinations
2. Allocate time budget according to the amount of combinations
3. Calculate number of possible tries based on time budget
4. Do n evaluations where the values are evenly spaced in the interval of [min, max] of the parameter
5. Keep the best value set out of this n tries