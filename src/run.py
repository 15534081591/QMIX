"""Ascend910run"""
import os
import pprint
import time
import threading
from types import SimpleNamespace as SN
from utils.logging import Logger
from utils.timehelper import time_left, time_str
from learners import REGISTRY as le_REGISTRY
from runners import REGISTRY as r_REGISTRY
from controllers import REGISTRY as mac_REGISTRY
from components.episode_buffer import ReplayBuffer
from components.transforms import OneHot
from envs import REGISTRY as env_REGISTRY
from mindspore import export, Tensor
import numpy as np


def run(_run, _config, _log):
    # check args sanity
    _config = args_sanity_check(_config, _log)

    args = SN(**_config)

    # setup loggers
    logger = Logger(_log)

    _log.info("Experiment Parameters:")
    experiment_params = pprint.pformat(_config,
                                       indent=4,
                                       width=1)
    _log.info("\n\n" + experiment_params + "\n")

    # sacred is on by default
    logger.setup_sacred(_run)

    # Run and train
    run_sequential(args=args, logger=logger)

    # Clean up after finishing
    print("Exiting Main")

    print("Stopping all threads")
    for t in threading.enumerate():
        if t.name != "MainThread":
            print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
            t.join(timeout=1)
            print("Thread joined")

    print("Exiting script")


def evaluate_sequential(args, runner):
    for _ in range(args.test_nepisode):
        runner.run(test_mode=True)
    runner.logger.log_stat("episode", 1, runner.t_env)
    runner.logger.print_recent_stats()
    if args.save_replay:
        runner.save_replay()

    runner.close_env()


def export_sequential(args, mac):
    if args.env_args['map_name'] == '3m':
        shape1 = [3, 42]
        shape2 = [3, 64]
    elif args.env_args['map_name'] == '2s3z':
        shape1 = [5, 96]
        shape2 = [5, 64]
    elif args.env_args['map_name'] == '5m_vs_6m':
        shape1 = [5, 72]
        shape2 = [5, 64]
    elif args.env_args['map_name'] == '3s5z':
        shape1 = [8, 150]
        shape2 = [8, 64]
    elif args.env_args['map_name'] == '8m':
        shape1 = [8, 102]
        shape2 = [8, 64]
    else:
        shape1 = [9, 186]
        shape2 = [9, 64]
    inputs = [Tensor(np.ones(shape1).astype(np.float32)), Tensor(np.ones(shape2).astype(np.float32))]
    export(mac.infer_agent, *inputs, file_name="agent", file_format="AIR")
    export(mac.infer_agent, *inputs, file_name="agent", file_format="MINDIR")
    print("export MINDIR file at {}".format("./agent.mindir"))
    print("export AIR file at {}".format("./agent.air"))



def run_sequential(args, logger):
    # Init runner so we can get env info
    runner = r_REGISTRY[args.runner](args=args, logger=logger)

    # Set up schemes and groups here
    env_info = runner.get_env_info()
    args.n_agents = env_info["n_agents"]
    args.n_actions = env_info["n_actions"]
    args.state_shape = env_info["state_shape"]

    # Default/Base scheme
    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": np.int32},
        "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": np.int32},
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": np.int32},
    }
    groups = {
        "agents": args.n_agents
    }
    preprocess = {
        "actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])
    }

    buffer = ReplayBuffer(scheme, groups, args.buffer_size, env_info["episode_limit"] + 1,
                          preprocess=preprocess)

    # Setup multiagent controller here
    mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)

    # Give runner the scheme
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)

    # Learner
    learner = le_REGISTRY[args.learner](mac, buffer.scheme, logger, args)

    if args.checkpoint_path != "":

        timesteps = []
        timestep_to_load = 0

        if not os.path.isdir(args.checkpoint_path):
            logger.console_logger.info(
                "Checkpoint directiory {} doesn't exist".format(args.checkpoint_path))
            return

        # Go through all files in args.checkpoint_path
        for name in os.listdir(args.checkpoint_path):
            full_name = os.path.join(args.checkpoint_path, name)
            # Check if they are dirs the names of which are numbers
            if os.path.isdir(full_name) and name.isdigit():
                timesteps.append(int(name))

        if args.load_step == 0:
            # choose the max timestep
            timestep_to_load = max(timesteps)
        else:
            # choose the timestep closest to load_step
            timestep_to_load = min(
                timesteps, key=lambda x: abs(x - args.load_step))

        model_path = os.path.join(args.checkpoint_path, str(timestep_to_load))

        logger.console_logger.info("Loading model from {}".format(model_path))
        learner.load_models(model_path)
        mac.load_infer_state()
        runner.t_env = timestep_to_load

        if args.evaluate or args.save_replay:
            evaluate_sequential(args, runner)
            return
        if args.export:
            export_sequential(args, mac)
            return
    # start training
    episode = 0
    last_test_T = -args.test_interval - 1
    last_log_T = 0
    model_save_time = 0

    start_time = time.time()
    last_time = start_time

    logger.console_logger.info(
        "Beginning training for {} timesteps".format(args.t_max))
    while runner.t_env <= args.t_max:
        # Run for a whole episode at a time
        mac.load_infer_state()
        episode_batch = runner.run(test_mode=False)
        buffer.insert_episode_batch(episode_batch)
        if buffer.can_sample(args.batch_size):
            step_start_time = time.time()
            episode_sample = buffer.sample(args.batch_size)
            learner.train(episode_sample, runner.t_env, episode)
            print("t_env: {} / {}".format(runner.t_env, args.t_max))
            print('step time:', (time.time() - step_start_time)*1000, "ms")
        # Execute test runs once in a while
        n_test_runs = max(1, args.test_nepisode // runner.batch_size)
        if (runner.t_env - last_test_T) / args.test_interval >= 1.0:

            logger.console_logger.info(
                "t_env: {} / {}".format(runner.t_env, args.t_max))
            logger.console_logger.info("Estimated time left: {}. Time passed: {}".format(
                time_left(last_time, last_test_T, runner.t_env, args.t_max), time_str(time.time() - start_time)))
            last_time = time.time()

            last_test_T = runner.t_env
            for _ in range(n_test_runs):
                runner.run(test_mode=True)

        if args.save_model and (runner.t_env - model_save_time >= args.save_model_interval or model_save_time == 0):
            model_save_time = runner.t_env
            save_path = os.path.join(
                args.local_results_path, "models", str(runner.t_env))
            # "results/models/{}".format(unique_token)
            os.makedirs(save_path, exist_ok=True)
            logger.console_logger.info("Saving models to {}".format(save_path))

            # learner should handle saving/loading -- delegate actor save/load to mac,
            # use appropriate filenames to do critics, optimizer states
            learner.save_models(save_path)
        episode += args.batch_size_run

        if (runner.t_env - last_log_T) >= args.log_interval:
            logger.log_stat("episode", episode, runner.t_env)
            logger.print_recent_stats()
            last_log_T = runner.t_env
            runner.env = env_REGISTRY[args.env](**args.env_args)

    logger.console_logger.info("Finished Training")


def args_sanity_check(config, _log):
    if config["test_nepisode"] < config["batch_size_run"]:
        config["test_nepisode"] = config["batch_size_run"]
    else:
        config["test_nepisode"] = (
            config["test_nepisode"] // config["batch_size_run"]) * config["batch_size_run"]

    return config
