import logging
import time
from dataclasses import dataclass

import yaml
from cost import cost
from param_types import TYPE_MAP, ParamType

logging.basicConfig(level=logging.INFO)


@dataclass
class ParamGroup:
    """
    A group with parameters that are interdependent to each other.
    """

    members: list[ParamType]

    def get_complexity_score(self) -> int:
        return sum(member.get_complexity_score() for member in self.members)

    def __str__(self):
        return f"ParamGroup({[member.name for member in self.members]})"


@dataclass
class SearchSpace:
    param_groups: list[ParamGroup]

    def build_start_config(self) -> dict[str, bool | float | int | str | list[int]]:
        return {
            member.name: member.val_range[0]
            for group in self.param_groups
            for member in group.members
        }


def load_params(file_name) -> SearchSpace:
    with open(file_name) as stream:
        try:
            params = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            raise SyntaxError("Error in the yaml file")
    all_parrams = {}
    for param in params["params"]:
        param_type_class = TYPE_MAP[param["type"]]
        args = {key: value for key, value in param.items() if key != "type"}
        all_parrams[param["name"]] = param_type_class(**args)
    search_space = SearchSpace([])
    for group in params["independent_groups"]:
        search_space.param_groups.append(
            ParamGroup([all_parrams[param] for param in group])
        )
    return search_space


def tune(params: SearchSpace, time_budget: int, time_budget_exponent: int = 1) -> float:
    """
    Tune the parameters in the search
    Args:
        params: SearchSpace with the parameters to tune
        time_budget: The time budget in seconds
        time_budget_exponent: The exponent for the time budget calculation for testing short time budgets

    Example: tune(params, 1, -2) will have a time budget of 1*10^-2 = 0.01 seconds
    """
    # Calculate total complexity and time budget
    total_complexity = sum(
        group.get_complexity_score() for group in params.param_groups
    )
    time_budget_factor = 10**time_budget_exponent
    time_budget = time_budget * time_budget_factor
    logging.info(f"Total time: {time_budget} seconds")
    # Test how long one evaluation takes
    t_start = time.time()
    param_config = params.build_start_config()
    eval_time = time.time() - t_start
    # Loop through the groups and tune the parameters of a group together
    for group in params.param_groups:
        max_score = float("-inf")
        best_group_config = {
            member.name: param_config[member.name] for member in group.members
        }
        # Calculate the time budget for the group and the number of runs
        group_time = (group.get_complexity_score() / total_complexity) * time_budget
        n_tries = int(group_time / eval_time)
        logging.info(f"Group {group} has {group_time} seconds and {n_tries} tries")
        # Quite rough tuning, just iterate over the value ranges of all params in the group
        for sample_set in zip(
            *(member.get_n_samples(n_tries) for member in group.members)
        ):
            # Update config with new values
            for param, value in zip(group.members, sample_set):
                param_config[param.name] = value
            # Evaluate the new config
            config_scoring = cost(param_config)
            # Save config if it beats the previous best
            if config_scoring > max_score:
                max_score = config_scoring
                best_group_config = {
                    member.name: param_config[member.name] for member in group.members
                }
            logging.debug(f"New best config: with score {max_score}")
        # Take the best group config to continue
        for param_name, value in best_group_config.items():
            param_config[param_name] = value
    logging.info(f"Best configuration: {param_config} with score {max_score}")
    logging.info(f"Time used: {time.time() - t_start}")
    return cost(param_config)


if __name__ == "__main__":
    import sys

    params = load_params(sys.argv[1])
    tune(params, int(sys.argv[2]), int(sys.argv[3]))
