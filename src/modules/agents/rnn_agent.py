"""train net"""
import numpy as np
import mindspore.ops as ops
from mindspore import nn


class RNNAgent(nn.Cell):
    def __init__(self, input_shape, args):
        super(RNNAgent, self).__init__()
        self.args = args
        self.fc1 = nn.Dense(input_shape, args.rnn_hidden_dim,
                            weight_init='uniform', bias_init='uniform')
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Dense(args.rnn_hidden_dim, args.n_actions,
                            weight_init='uniform', bias_init='uniform')
        self.relu = ops.ReLU()
        self.reshape = ops.Reshape()
        self.h_in_shape = self.args.rnn_hidden_dim
        self.batch_size = self.args.batch_size
        self.n_agents = self.args.n_agents

    def init_hidden(self):
        # make hidden states on same device as model
        return np.zeros([1, self.args.rnn_hidden_dim], np.float32)

    def reshape_hidden(self, hidden_state):
        h_in = self.reshape(hidden_state, (-1, self.h_in_shape))
        return h_in

    def construct(self, inputs, h_in, max_seq_length):
        mac_out = []
        for t in range(max_seq_length):
            if t == 0:
                h_in = self.reshape_hidden(h_in)
            fc1_out = self.fc1(inputs[t])
            x = self.relu(fc1_out)
            h_in = self.rnn(x, h_in)
            q = self.fc2(h_in)
            mac_out.append(q.view(self.batch_size, self.n_agents, -1))
        return mac_out
