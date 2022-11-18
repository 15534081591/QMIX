"""Ascend310run"""
import os
import pprint
import time
from types import SimpleNamespace as SN
from utils.logging import Logger
from envs import REGISTRY as env_REGISTRY
from components.episode_buffer import EpisodeBatch
from components.transforms import OneHot
from components.action_selectors import REGISTRY as action_REGISTRY
import numpy as np


def run_310(_run, _config, _log):
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
    env = env_REGISTRY[args.env](**args.env_args)
    env_info = env.get_env_info()
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
    test_stats = {}
    test_returns = []
    for _ in range(args.test_nepisode):
        batch = EpisodeBatch(scheme, groups, args.batch_size_run, env.episode_limit + 1,
                             preprocess=preprocess)
        action_selector = action_REGISTRY[args.action_selector](args)
        env.reset()
        t = 0
        t_env = 0
        terminated = False
        episode_return = 0
        tmp = np.expand_dims(np.zeros([1, args.rnn_hidden_dim], np.float32), axis=0)
        shape = tmp.shape
        hidden_states = np.broadcast_to(tmp, (args.batch_size_run, args.n_agents, shape[2]))
        while not terminated:
            start_time = time.time()
            pre_transition_data = {
                "state": [env.get_state()],
                "avail_actions": [env.get_avail_actions()],
                "obs": [env.get_obs()]
            }
            batch.update(pre_transition_data, ts=t)
            actions, hidden_states = select_actions(
                batch, t, t_env, args, hidden_states, action_selector)
            reward, terminated, env_info = env.step(actions[0])
            episode_return += reward
            post_transition_data = {
                "actions": actions,
                "reward": [(reward,)],
                "terminated": [(terminated != env_info.get("episode_limit", False),)],
            }
            batch.update(post_transition_data, ts=t)
            t += 1
            print('step time:', (time.time() - start_time)*1000, "ms")
        last_data = {
            "state": [env.get_state()],
            "avail_actions": [env.get_avail_actions()],
            "obs": [env.get_obs()]
        }
        batch.update(last_data, ts=t)
        actions, hidden_states = select_actions(
            batch, t, t_env, args, hidden_states, action_selector)
        batch.update({"actions": actions}, ts=t)
        cur_stats = test_stats
        cur_returns = test_returns
        log_prefix = "test_"
        cur_stats.update({k: cur_stats.get(k, 0) + env_info.get(k, 0)
                          for k in set(cur_stats) | set(env_info)})
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = t + cur_stats.get("ep_length", 0)
        cur_returns.append(episode_return)
        if len(test_returns) == args.test_nepisode:
            logger.log_stat(log_prefix + "return_mean",
                            np.mean(cur_returns), t_env)
            logger.log_stat(log_prefix + "return_std",
                            np.std(cur_returns), t_env)
            cur_returns.clear()

            for k, v in cur_stats.items():
                if k != "n_episodes":
                    logger.log_stat(log_prefix + k + "_mean",
                                    v / cur_stats["n_episodes"], t_env)
            cur_stats.clear()
    env.close()
    logger.log_stat("episode", 1, t_env)
    logger.print_recent_stats()

def select_actions(batch, t, t_env, args, hidden_states, action_selector):
    avail_actions = batch["avail_actions"][:, t]
    agent_inputs = build_inputs(batch, t, args)
    shape = hidden_states.shape
    hidden_states = np.broadcast_to(hidden_states, (args.batch_size_run, args.n_agents, shape[-1]))
    h_in = np.reshape(hidden_states, (-1, args.rnn_hidden_dim))
    result_path = os.path.join(args.output_path)
    agent_inputs.tofile(os.path.join(result_path, "agent_inputs.bin"))
    h_in.tofile(os.path.join(result_path, "h_in.bin"))
    print(args.run_main)
    os.system("{} --model_path={} --output_path={} --device_id={}".format(args.run_main,
                                                                          args.model_path,
                                                                          args.output_path,
                                                                          args.device_id))
    agent_outs = np.fromfile(os.path.join(
        result_path, "outputs_0.bin"), dtype=np.float32)
    agent_outs = agent_outs.reshape((agent_inputs.shape[0], agent_outs.shape[0] // agent_inputs.shape[0]))
    hidden_states = np.fromfile(os.path.join(
        result_path, "outputs_1.bin"), dtype=np.float32)
    hidden_states = hidden_states.reshape(h_in.shape)
    agent_outs = agent_outs.reshape(batch.batch_size, args.n_agents, -1)
    actions = action_selector.select_action(agent_outs[slice(None)], avail_actions[slice(None)], t_env,
                                            test_mode=True)
    return actions, hidden_states


def build_inputs(batch, t, args):
    # Assumes homogeneous agents with flat observations.
    # Other MACs might want to e.g. delegate building inputs to each agent
    bs = batch.batch_size
    inputs = []
    inputs.append(batch["obs"][:, t])  # b1av
    if args.obs_last_action:
        if t == 0:
            inputs.append(np.zeros_like(batch["actions_onehot"][:, t]))
        else:
            inputs.append(batch["actions_onehot"][:, t - 1])
    if args.obs_agent_id:
        tmp = np.expand_dims(np.eye(args.n_agents, args.n_agents, dtype=np.float32), axis=0)
        shape = tmp.shape
        inputs.append(np.broadcast_to(tmp, (bs, shape[1], shape[2])))
    inputs = np.concatenate([x.reshape((bs * args.n_agents, -1)) for x in inputs], axis=1)
    return inputs
