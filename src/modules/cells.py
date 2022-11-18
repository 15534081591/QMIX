"""netWithLossCell"""
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.ops.composite as C
from mindspore.context import ParallelMode
from mindspore.ops import functional as F
from mindspore.ops import stop_gradient
from mindspore.parallel._utils import (_get_device_num, _get_gradients_mean,
                                       _get_parallel_mode)
from .grad_clip import GRADIENT_CLIP_TYPE, clip_grad


class netWithLossCell(nn.Cell):
    """Generator with loss(wrapped)"""

    def __init__(self, args, mac, target_mac, mixer, target_mixer):
        super(netWithLossCell, self).__init__()
        self.mixer = mixer
        self.target_mixer = target_mixer
        self.agent = mac.agent
        self.target_agent = target_mac.agent
        self.n_agents = mac.n_agents
        self.target_n_agents = target_mac.n_agents
        self.batch_size = args.batch_size
        self.double_q = args.double_q
        self.gamma = args.gamma
        self.stack = ops.Stack(axis=1)
        self.gather = ops.GatherD()
        self.max = ops.ArgMaxWithValue(axis=3, keep_dims=True)
        self.max_not_dims = ops.ArgMaxWithValue(axis=3, keep_dims=False)
        self.squeeze = ops.Squeeze(3)

    def construct(self, rewards, actions, terminated, mask, avail_actions, state, max_seq_length, agent_inputs,
                  target_agent_inputs, hidden_states, target_hidden_states):

        mac_out = self.agent(agent_inputs, hidden_states, max_seq_length)
        mac_out = self.stack(mac_out)
        chosen_action_qvals = self.squeeze(self.gather(mac_out[:, :-1], 3, actions))
        target_mac_out = self.target_agent(target_agent_inputs, target_hidden_states, max_seq_length)
        target_mac_out = self.stack(target_mac_out)[:, 1:]
        target_mac_out[avail_actions[:, 1:] == 0] = -9999999
        if self.double_q:
            mac_out_detach = stop_gradient(mac_out)
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = self.max(mac_out_detach[:, 1:])[0]
            target_max_qvals = self.squeeze(self.gather(target_mac_out, 3, cur_max_actions))
        else:
            target_max_qvals = self.max_not_dims(target_mac_out)[1]
        if self.mixer is not None:
            chosen_action_qvals = self.mixer(chosen_action_qvals, state[:, :-1])
            target_max_qvals = self.target_mixer(target_max_qvals, state[:, 1:])
        targets = rewards + self.gamma * (1 - terminated) * target_max_qvals
        td_error = (chosen_action_qvals - stop_gradient(targets))
        mask = mask.expand_as(td_error)
        masked_td_error = td_error * mask
        loss = (masked_td_error ** 2).sum() / mask.sum()
        return loss


class QmixTrainOneStepCell(nn.Cell):
    def __init__(self, args, network, optimizer, sens=1.0):
        super(QmixTrainOneStepCell, self).__init__(auto_prefix=False)
        self.network = network
        self.grad_norm_clip = args.grad_norm_clip
        self.network.set_grad()
        self.optimizer = optimizer
        self.weights = self.optimizer.parameters
        self.grad = C.GradOperation(get_by_list=True, sens_param=True)
        self.sens = sens
        self.reducer_flag = False
        self.grad_reducer = F.identity
        self.parallel_mode = _get_parallel_mode()
        self.reducer_flag = self.parallel_mode in (ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL)
        if self.reducer_flag:
            self.mean = _get_gradients_mean()
            self.degree = _get_device_num()
            self.grad_reducer = nn.DistributedGradReducer(self.weights, self.mean, self.degree)
        self.hyper_map = C.HyperMap()

    def construct(self, rewards, actions, terminated, mask, avail_actions, state,
                  max_seq_length,
                  agent_inputs, target_agent_inputs, hidden_states,
                  target_hidden_states):
        weights = self.weights
        loss = self.network(rewards, actions, terminated, mask, avail_actions, state,
                            max_seq_length,
                            agent_inputs, target_agent_inputs, hidden_states,
                            target_hidden_states)
        sens = F.fill(loss.dtype, loss.shape, self.sens)
        grads = self.grad(self.network, weights)(rewards, actions, terminated, mask, avail_actions, state,
                                                 max_seq_length,
                                                 agent_inputs, target_agent_inputs, hidden_states,
                                                 target_hidden_states, sens)
        grads = self.hyper_map(F.partial(clip_grad, GRADIENT_CLIP_TYPE, self.grad_norm_clip), grads)
        if self.reducer_flag:
            # apply grad reducer on grads
            grads = self.grad_reducer(grads)
        return F.depend(loss, self.optimizer(grads))
