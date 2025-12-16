from collections.abc import Callable
import math

from interopt import Study
from interopt.parameter import (
    Boolean as BooleanInteropt,
    Categorical as CategoricalInteropt,
    Integer as IntegerInteropt,
    IntExponential as IntExponentialInteropt,
    Permutation as PermutationInteropt,
    Real as RealInteropt,
)

from schedgehammer.problem import Problem
from schedgehammer.param_types import Param
from schedgehammer.param_types import (
    CategoricalParam,
    ExpIntParam,
    IntegerParam,
    PermutationParam,
    RealParam,
    SwitchParam,
)


class CatbenchProblem(Problem):
    def __init__(
        self,
        name: str,
        params: dict[str, Param],
        study: Study,
        constraints: list[str] = [],
    ):
        super().__init__(name, params, constraints, init_solver=True)
        self.fidelity_params = {}
        for fidelity_param in study.definition.search_space.fidelity_params:
            self.fidelity_params[fidelity_param.name] = fidelity_param.default
        self.study = study

    def cost_function(self, config: dict[str, any]) -> float:
        config = config.copy()
        for name, val in config.items():
            if type(val) is list:
                config[name] = str(val)

        # print(f"query: {config}")
        # print(f"fidelyti param: {self.fidelity_params}")
        # print("query")
        result = self.study.query(config, self.fidelity_params)["compute_time"]
        # print(f"query result: {result}")

        return result


def problem_from_study(study: Study) -> Problem:
    params = {}
    for param in study.definition.search_space.params:
        if type(param) is CategoricalInteropt:
            params[param.name] = CategoricalParam(
                values=param.categories,
            )
        elif type(param) is PermutationInteropt:
            params[param.name] = PermutationParam(values=list(param.default))
        elif type(param) is BooleanInteropt:
            params[param.name] = SwitchParam()
        elif type(param) is IntegerInteropt:
            params[param.name] = IntegerParam(
                min_val=param.bounds[0],
                max_val=param.bounds[1],
            )
        elif type(param) is RealInteropt:
            params[param.name] = RealParam(
                min_val=param.bounds[0],
                max_val=param.bounds[1],
            )
        elif type(param) is IntExponentialInteropt:
            params[param.name] = ExpIntParam(
                base=param.base,
                min_exp=math.log(param.bounds[0], param.base),
                max_exp=math.log(param.bounds[1], param.base),
            )
        else:
            raise ValueError(f"Problem got unsupported parameter type: {type(param)}")

    # def interop_eval(config):
    #     config = config.copy()
    #     for name, val in config.items():
    #         if type(val) is list:
    #             config[name] = str(val)
    #     return study.query(config, fidelity_params)["compute_time"]

    return CatbenchProblem(
        study.definition.name,
        params,
        study,
        [c.constraint for c in study.definition.search_space.constraints],
    )


def pyatf_from_study(study: Study):
    """
    Translate catbench Study to pyATF tuning parameters.
    
    Returns:
        tuple: (list of pyATF TP objects, cost_function)
    """
    try:
        from pyatf import TP, Interval, Set
        from pyatf.tuning_data import Cost, CostFunctionError
        import itertools
    except ImportError:
        raise ImportError("pyATF is not installed. Install it with: pip install pyatf")
    
    # Collect fidelity parameters
    fidelity_params = {}
    for fidelity_param in study.definition.search_space.fidelity_params:
        fidelity_params[fidelity_param.name] = fidelity_param.default
    
    # Build pyATF tuning parameters
    tuning_params = []
    
    for param in study.definition.search_space.params:
        if type(param) is CategoricalInteropt:
            # Use pyATF's Set for categorical values
            tp = TP(
                param.name,
                Set(*param.categories)
            )
            tuning_params.append(tp)
            
        elif type(param) is PermutationInteropt:
            # Use pyATF's Set with all possible permutations
            all_permutations = list(itertools.permutations(param.default))
            # Convert tuples to lists for JSON serialization compatibility
            all_permutations = [list(p) for p in all_permutations]
            tp = TP(
                param.name,
                Set(*all_permutations)
            )
            tuning_params.append(tp)
            
        elif type(param) is BooleanInteropt:
            # Boolean as 0/1 integer
            tp = TP(
                param.name,
                Interval(0, 1),
                lambda **kwargs: kwargs.get(param.name, 0) in [0, 1]
            )
            tuning_params.append(tp)
            
        elif type(param) is IntegerInteropt:
            # Direct integer interval
            tp = TP(
                param.name,
                Interval(param.bounds[0], param.bounds[1])
            )
            tuning_params.append(tp)
            
        elif type(param) is RealInteropt:
            # Real interval (pyATF supports real-valued parameters)
            tp = TP(
                param.name,
                Interval(param.bounds[0], param.bounds[1])
            )
            tuning_params.append(tp)
            
        elif type(param) is IntExponentialInteropt:
            # Exponential integer: use integer interval and apply transformation in cost function
            # Store bounds in log space
            min_exp = int(math.log(param.bounds[0], param.base))
            max_exp = int(math.log(param.bounds[1], param.base))
            tp = TP(
                param.name,
                Interval(min_exp, max_exp)
            )
            # Mark as exponential for cost function transformation
            tp._catbench_exponential = True
            tp._catbench_base = param.base
            tuning_params.append(tp)
            
        else:
            raise ValueError(f"Unsupported parameter type: {type(param)}")
    
    # Add constraints from catbench
    # pyATF constraints are expressed as lambda functions in TP definitions
    # We need to translate string constraints to lambda functions
    # This is complex - constraints in catbench are string expressions
    # For now, we'll add them as a post-processing step
    
    # Note: catbench constraints are string-based expressions like "x < y"
    # pyATF constraints are lambda functions like: lambda x, y: x < y
    # This translation requires parsing the constraint string and converting to lambda
    # For simplicity, we'll leave this as a TODO or manual step
    
    if study.definition.search_space.constraints:
        print(f"Warning: catbench has {len(study.definition.search_space.constraints)} constraints.")
        print("Constraint translation from string to lambda is not yet automated.")
        print("You may need to manually add constraints to pyATF tuning parameters.")
        print("Constraints:")
        for c in study.definition.search_space.constraints:
            print(f"  - {c.constraint}")
    
    # Create cost function
    def cost_function(**config):
        """Cost function that queries catbench with the configuration."""
        # Transform exponential parameters back
        transformed_config = {}
        for param_name, value in config.items():
            # Check if this is an exponential parameter that needs transformation
            for tp in tuning_params:
                if tp.name == param_name:
                    if hasattr(tp, '_catbench_exponential') and tp._catbench_exponential:
                        # Convert from log space to actual value
                        value = tp._catbench_base ** value
                    break
            
            # For Set parameters (categorical and permutation), value is already correct
            # For permutations that come as lists, convert to string format if needed
            if isinstance(value, list):
                # Check if catbench expects string representation
                transformed_config[param_name] = str(value)
            else:
                transformed_config[param_name] = value
        
        try:
            # Query catbench
            result = study.query(transformed_config, fidelity_params)["compute_time"]
            return Cost(result)
        except Exception as e:
            # If query fails, raise CostFunctionError to penalize this configuration
            raise CostFunctionError(f"Configuration failed: {e}")
    
    return tuning_params, cost_function
