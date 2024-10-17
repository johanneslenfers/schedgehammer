import logging
from dataclasses import dataclass

import yaml
from param_types import TYPE_MAP, ParamType


@dataclass
class ParamGroup:
    """
    A group with parameters that are interdependent to each other.
    """

    members: list[ParamType]

    def get_complexity_score(self) -> int:
        return sum(member.get_complexity_score() for member in self.members)


@dataclass
class ParamConfig:
    param_groups: list[ParamGroup]


def load_params(file_name) -> ParamConfig:
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
    param_config = ParamConfig([])
    for group in params["independent_groups"]:
        param_config.param_groups.append(
            ParamGroup([all_parrams[param] for param in group])
        )
    return param_config


def tune(params: ParamConfig, time_budget: int, time_budget_factor: int = 1):
    total_complexity = sum(
        group.get_complexity_score() for group in params.param_groups
    )
    for group in params.param_groups:
        group_time = (
            (group.get_complexity_score() / total_complexity)
            * time_budget
            * time_budget_factor
        )

        logging.info(f"Group {group} has {group_time} seconds")
