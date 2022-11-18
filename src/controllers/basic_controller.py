"""controllers"""
from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.train.serialization import save_checkpoint
from mindspore.train.serialization import load_checkpoint, load_param_into_net
import numpy as np



# This multi-agent controller shares parameters between agents
class BasicMAC(nn.Cell):
    def __init__(self, scheme, groups, args):
        super(BasicMAC, self).__init__()

        self.n_agents = args.n_agents
        self.args = args
        input_shape = self._get_input_shape(scheme)
        self._build_agents(input_shape)
        self.agent_output_type = args.agent_output_type

        self.action_selector = action_REGISTRY[args.action_selector](args)
        self.hidden_states = None

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_outputs = self.forward(ep_batch, t_ep, test_mode=test_mode)
        chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env,
                                                            test_mode=test_mode)
        return chosen_actions

    def forward(self, ep_batch, t, test_mode=False):
        agent_inputs = self.build_inputs(ep_batch, t)
        shape = self.hidden_states.shape
        self.hidden_states = np.broadcast_to(self.hidden_states, (self.args.batch_size_run, self.n_agents, shape[-1]))
        self.hidden_states = self.infer_agent.reshape_hidden(self.hidden_states)
        agent_inputs = Tensor(agent_inputs, ms.float32)
        hidden_states = Tensor(self.hidden_states, ms.float32)
        agent_outs, self.hidden_states = self.infer_agent(agent_inputs, hidden_states)
        agent_outs = agent_outs.asnumpy()
        self.hidden_states = self.hidden_states.asnumpy()
        return agent_outs.reshape(ep_batch.batch_size, self.n_agents, -1)


    def init_hidden(self, batch_size):
        tmp = np.expand_dims(self.agent.init_hidden(), axis=0)
        shape = tmp.shape
        self.hidden_states = np.broadcast_to(tmp, (batch_size, self.n_agents, shape[2]))

    def parameters(self):
        return self.agent.trainable_params()

    def load_infer_state(self):
        par_dict = self.agent.parameters_dict()
        param_dict = {}
        for name in par_dict:
            parameter = par_dict[name]
            name = name.replace('qmix.netWithLoss.agent.', 'mac.infer_agent.', 1)
            param_dict[name] = parameter
        load_param_into_net(self.infer_agent, param_dict)

    def load_state(self, other_mac):
        par_dict = other_mac.agent.parameters_dict()
        param_dict = {}
        for name in par_dict:
            parameter = par_dict[name]
            name = name.replace('netWithLoss.agent.', 'network.target_agent.', 1)
            param_dict[name] = parameter
        load_param_into_net(self.agent, param_dict)

    def save_models(self, path):
        save_checkpoint(self.agent, "{}/agent.ckpt".format(path))

    def load_models(self, path):
        param_dict = load_checkpoint("{}/agent.ckpt".format(path))
        load_param_into_net(self.agent, param_dict)

    def load_target_models(self, path):
        par_dict = load_checkpoint("{}/agent.ckpt".format(path))
        param_dict = {}
        for name in par_dict:
            parameter = par_dict[name]
            name = name.replace('agent.', 'target_agent.', 1)
            param_dict[name] = parameter
        load_param_into_net(self.agent, param_dict)

    def _build_agents(self, input_shape):
        self.agent = agent_REGISTRY[self.args.agent](input_shape, self.args)
        self.infer_agent = agent_REGISTRY['infer'](input_shape, self.args)

    def build_inputs(self, batch, t):
        # Assumes homogeneous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs"][:, t])  # b1av
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(np.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t - 1])
        if self.args.obs_agent_id:
            tmp = np.expand_dims(np.eye(self.n_agents, self.n_agents, dtype=np.float32), axis=0)
            shape = tmp.shape
            inputs.append(np.broadcast_to(tmp, (bs, shape[1], shape[2])))
        inputs = np.concatenate([x.reshape((bs * self.n_agents, -1)) for x in inputs], axis=1)
        return inputs


    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents

        return input_shape
