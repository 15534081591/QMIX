"""mix net"""
import numpy as np
import mindspore.nn as nn
import mindspore.ops as ops

class QMixer(nn.Cell):
    def __init__(self, args):
        super(QMixer, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.state_dim = int(np.prod(args.state_shape))

        self.embed_dim = args.mixing_embed_dim

        if getattr(args, "hypernet_layers", 1) == 1:
            self.hyper_w_1 = nn.Dense(self.state_dim, self.embed_dim * self.n_agents, weight_init='uniform',
                                      bias_init='uniform')
            self.hyper_w_final = nn.Dense(self.state_dim, self.embed_dim, weight_init='uniform', bias_init='uniform')
        elif getattr(args, "hypernet_layers", 1) == 2:
            hypernet_embed = self.args.hypernet_embed
            self.hyper_w_1 = nn.SequentialCell(
                nn.Dense(self.state_dim, hypernet_embed, weight_init='uniform', bias_init='uniform'),
                nn.ReLU(),
                nn.Dense(hypernet_embed, self.embed_dim * self.n_agents, weight_init='uniform', bias_init='uniform'))
            self.hyper_w_final = nn.SequentialCell(
                nn.Dense(self.state_dim, hypernet_embed, weight_init='uniform', bias_init='uniform'),
                nn.ReLU(),
                nn.Dense(hypernet_embed, self.embed_dim, weight_init='uniform', bias_init='uniform'))
        elif getattr(args, "hypernet_layers", 1) > 2:
            raise Exception("Sorry >2 hypernet layers is not implemented!")
        else:
            raise Exception("Error setting number of hypernet layers.")

        # State dependent bias for hidden layer
        self.hyper_b_1 = nn.Dense(self.state_dim, self.embed_dim, weight_init='uniform', bias_init='uniform')

        # V(s) instead of a bias for the last layers
        self.V = nn.SequentialCell(nn.Dense(self.state_dim, self.embed_dim, weight_init='uniform', bias_init='uniform'),
                                   nn.ReLU(),
                                   nn.Dense(self.embed_dim, 1, weight_init='uniform', bias_init='uniform'))
        self.reshape = ops.Reshape()
        self.abs = ops.Abs()
        self.elu = ops.Elu()
        self.batmatmul = ops.BatchMatMul()

    #@ms_function
    def construct(self, agent_qs, states):
        bs = agent_qs.shape[0]
        states = self.reshape(states, (-1, self.state_dim))
        agent_qs = agent_qs.view((-1, 1, self.n_agents))
        w1 = self.abs(self.hyper_w_1(states))
        b1 = self.hyper_b_1(states)
        w1 = w1.view((-1, self.n_agents, self.embed_dim))
        b1 = b1.view((-1, 1, self.embed_dim))
        hidden = self.elu(self.batmatmul(agent_qs, w1) + b1)
        w_final = self.abs(self.hyper_w_final(states))
        w_final = w_final.view((-1, self.embed_dim, 1))
        v = self.V(states).view((-1, 1, 1))
        y = self.batmatmul(hidden, w_final) + v
        q_tot = y.view((bs, -1, 1))
        return q_tot
