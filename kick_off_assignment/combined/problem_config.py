from dataclasses import dataclass

import yaml
from param_types import TYPE_MAP, Param


@dataclass
class ProblemConfig:
    params: dict[str, Param]

    def to_dict(self) -> dict:
        return {name: param.val for name, param in self.params.items()}

    @classmethod
    def from_file(cls, file_name: str):
        with open(file_name) as stream:
            try:
                params = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
                raise SyntaxError("Error in the yaml file")
        all_params = {}
        for param in params["params"]:
            param_type_class = TYPE_MAP[param["type"]]
            args = {key: value for key, value in param.items() if key != "type"}
            all_params[param["name"]] = param_type_class(**args)
        return cls(all_params)
