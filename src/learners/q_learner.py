"""learner"""
import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.qmix import QMixer
from modules.cells import netWithLossCell, QmixTrainOneStepCell
from mindspore import nn
from mindspore import load_checkpoint, load_param_into_net
from mindspore.train.serialization import save_checkpoint
from mindspore import Tensor
import mindspore.ops as ops
import mindspore as ms
import numpy as np



class QLearner(nn.Cell):
    def __init__(self, mac, scheme, logger, args):
        super(QLearner, self).__init__()
        self.args = args
        self.mac = mac
        self.logger = logger

        self.params = list(mac.parameters())

        self.last_target_update_episode = 0

        self.mixer = None
        if args.mixer is not None:
            if args.mixer == "qmix":
                self.mixer = QMixer(args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
            self.params += list(self.mixer.trainable_params())
            self.target_mixer = copy.deepcopy(self.mixer)

        self.optimiser = nn.RMSProp(params=self.params, learning_rate=args.lr, decay=args.optim_alpha,
                                    epsilon=args.optim_eps)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)
        self.netWithLoss = netWithLossCell(self.args, self.mac, self.target_mac, self.mixer, self.target_mixer)
        self.qmix = QmixTrainOneStepCell(self.args, self.netWithLoss, self.optimiser)
        self.log_stats_t = -self.args.learner_log_interval - 1
        self.stack_0 = ops.Stack(axis=0)

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].astype('float32')
        mask = batch["filled"][:, :-1].astype('float32')
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]
        state = batch["state"]
        self.mac.init_hidden(batch.batch_size)
        agent_inputs = []
        for t in range(batch.max_seq_length):
            agent_input = self.mac.build_inputs(batch, t=t)
            agent_inputs.append(agent_input)
        target_agent_inputs = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_input = self.target_mac.build_inputs(batch, t=t)
            target_agent_inputs.append(target_agent_input)
        agent_inputs = np.stack(agent_inputs, axis=0)
        target_agent_inputs = np.stack(target_agent_inputs, axis=0)
        hidden_states = self.mac.hidden_states
        target_hidden_states = self.target_mac.hidden_states
        rewards = Tensor(rewards, ms.float32)
        actions = Tensor(actions, ms.int32)
        terminated = Tensor(terminated, ms.float32)
        mask = Tensor(terminated, ms.float32)
        avail_actions = Tensor(avail_actions, ms.int32)
        state = Tensor(state, ms.float32)
        agent_inputs = Tensor(agent_inputs, ms.float32)
        target_agent_inputs = Tensor(target_agent_inputs, ms.float32)
        hidden_states = Tensor(hidden_states, ms.float32)
        target_hidden_states = Tensor(target_hidden_states, ms.float32)
        loss = self.qmix(rewards, actions, terminated, mask, avail_actions, state,
                         batch.max_seq_length,
                         agent_inputs, target_agent_inputs, hidden_states,
                         target_hidden_states)
        print(loss)

        self.mac.load_infer_state()
        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.asnumpy(), t_env)
            self.log_stats_t = t_env

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            par_dict = self.mixer.parameters_dict()
            param_dict = {}
            for name in par_dict:
                parameter = par_dict[name]
                name = name.replace('netWithLoss.mixer.', 'network.target_mixer.', 1)
                param_dict[name] = parameter
            load_param_into_net(self.target_mixer, param_dict)
        self.logger.console_logger.info("Updated target network")

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            save_checkpoint(self.mixer, "{}/mixer.ckpt".format(path))
        save_checkpoint(self.optimiser, "{}/opt.ckpt".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_target_models(path)
        if self.mixer is not None:
            param_dict = load_checkpoint("{}/mixer.ckpt".format(path))
            load_param_into_net(self.mixer, param_dict)
        param_dict = load_checkpoint("{}/opt.ckpt".format(path))
        load_param_into_net(self.optimiser, param_dict)
