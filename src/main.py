import collections
import os
import sys
from copy import deepcopy
from logging import Logger
from os.path import abspath, dirname
from typing import Any, Dict, List, Optional

import numpy as np
import torch as th
import yaml
from sacred import SETTINGS, Experiment
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds
from sacred.run import Run

from run import run
from utils.logging import get_logger

SETTINGS[
    "CAPTURE_MODE"
] = "fd"  # set to "no" if you want to see stdout/stderr in console
logger = get_logger()

ex = Experiment("pymarl")
ex.logger = logger
ex.captured_out_filter = apply_backspaces_and_linefeeds

results_path = os.path.join(dirname(dirname(abspath(__file__))), "results")


@ex.main
def my_main(_run: Run, _config: Dict[str, Any], _log: Logger):
    # Setting the random seed throughout the modules
    config = config_copy(_config)
    np.random.seed(config["seed"])
    th.manual_seed(config["seed"])
    config["env_args"]["seed"] = config["seed"]

    # run the framework
    run(_run, config, _log)


def _get_config(params: List[str], arg_name: str, subfolder: str) -> Dict[str, Any]:
    """Get config

    Args:
        params (List[str]): arguments
        arg_name (str): argument "env_config"
        subfolder (str): subfolder "envs"

    Returns:
        _type_: _description_
    """
    config_name: Optional[str] = None
    for _i, _v in enumerate(params):
        if _v.split("=")[0] == arg_name:
            config_name = _v.split("=")[1]
            del params[_i]
            break

    if config_name is not None:
        with open(
            os.path.join(
                os.path.dirname(__file__),
                "config",
                subfolder,
                "{}.yaml".format(config_name),
            ),
            "r",
        ) as f:
            try:
                config_dict = yaml.safe_load(f)
            except yaml.YAMLError as exc:
                assert False, "{}.yaml error: {}".format(config_name, exc)
        return config_dict


def recursive_dict_update(d: Dict[str, Any], u: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in u.items():
        if isinstance(v, collections.ChainMap):
            d[k] = recursive_dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def config_copy(config):
    if isinstance(config, dict):
        return {k: config_copy(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [config_copy(v) for v in config]
    else:
        return deepcopy(config)


if __name__ == "__main__":
    params = deepcopy(sys.argv)

    # Get the defaults from default.yaml
    with open(
        os.path.join(os.path.dirname(__file__), "config", "default.yaml"), "r"
    ) as f:
        try:
            config_dict = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            assert False, "default.yaml error: {}".format(exc)

    # Load algorithm and env base configs
    env_config = _get_config(params, "--env-config", "envs")
    alg_config = _get_config(params, "--config", "algs")
    # config_dict = {**config_dict, **env_config, **alg_config}
    config_dict = recursive_dict_update(config_dict, env_config)
    config_dict = recursive_dict_update(config_dict, alg_config)

    # now add all the config to sacred
    ex.add_config(config_dict)

    # Save to disk by default for sacred
    logger.info("Saving to FileStorageObserver in results/sacred.")
    file_obs_path = os.path.join(results_path, "sacred")
    ex.observers.append(FileStorageObserver.create(file_obs_path))

    ex.run_commandline(params)
