"""main"""
import os
from os.path import dirname, abspath
import collections
from copy import deepcopy
import sys
import yaml
import numpy as np
from sacred import Experiment, SETTINGS
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds
from utils.logging import get_logger
from mindspore import context
from run import run
from run_310 import run_310


SETTINGS['CAPTURE_MODE'] = "fd"
logger = get_logger()

ex = Experiment("pymarl")
ex.logger = logger
ex.captured_out_filter = apply_backspaces_and_linefeeds

results_path = os.path.join(dirname(dirname(abspath(__file__))), "results")


@ex.main
def my_main(_run, _config, _log):
    # Setting the random seed throughout the modules
    config = config_copy(_config)
    np.random.seed(config["seed"])
    config['env_args']['seed'] = config["seed"]
    # run the framework
    if config['evaluate_310']:
        context.set_context(device_target="CPU", mode=context.PYNATIVE_MODE)
        run_310(_run, config, _log)
    else:
        context.set_context(device_target="Ascend",
                            device_id=config['device_id'], mode=context.GRAPH_MODE)
        run(_run, config, _log)


def get_config(param, arg_name, subfolder):
    config_name = None
    for _i, _v in enumerate(param):
        if _v.split("=")[0] == arg_name:
            config_name = _v.split("=")[1]
            del param[_i]
            break

    if config_name is not None:
        with open(os.path.join(os.path.dirname(__file__), "config", subfolder, "{}.yaml".format(config_name)),
                  "r") as file:
            try:
                config_dict_ = yaml.safe_load(file)
            except yaml.YAMLError as exc:
                assert False, "{}.yaml error: {}".format(config_name, exc)
        return config_dict_
    return None


def recursive_dict_update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            d[k] = recursive_dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def config_copy(config):
    if isinstance(config, dict):
        return {k: config_copy(v) for k, v in config.items()}
    if isinstance(config, list):
        return [config_copy(v) for v in config]
    return deepcopy(config)


if __name__ == '__main__':
    params = deepcopy(sys.argv)
    if len(params) == 1:
        params.append("--config=qmix")
        params.append('--env-config=sc2')
        params.append('with')
        params.append('env_args.map_name=3m')
    with open(os.path.join(os.path.dirname(__file__), "config", "default.yaml"), "r") as f:
        try:
            config_dict = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            assert False, "default.yaml error: {}".format(exc)

    # Load algorithm and env base configs
    env_config = get_config(params, "--env-config", "envs")
    alg_config = get_config(params, "--config", "algs")
    config_dict = recursive_dict_update(config_dict, env_config)
    config_dict = recursive_dict_update(config_dict, alg_config)

    # now add all the config to sacred
    ex.add_config(config_dict)

    # Save to disk by default for sacred
    logger.info("Saving to FileStorageObserver in results/sacred.")
    file_obs_path = os.path.join(results_path, "sacred")
    ex.observers.append(FileStorageObserver.create(file_obs_path))

    ex.run_commandline(params)
