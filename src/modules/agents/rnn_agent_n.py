import torch.nn as nn
import torch.nn.functional as F

class agent_ensemble_network(nn.Module):
    def __init__(self, input_shape, args):  
        super(agent_ensemble_network, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)


    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()
        
    def forward(self, inputs, hidden_state):
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        return q, h

class RNNAgentN(nn.Module):
    def __init__(self, input_shape, args):
        super(RNNAgentN, self).__init__()
        self.args = args

        self.network1 = agent_ensemble_network(input_shape, self.args)
        self.network2 = agent_ensemble_network(input_shape, self.args)
        self.network3 = agent_ensemble_network(input_shape, self.args)
        self.network4 = agent_ensemble_network(input_shape, self.args)
        self.network5 = agent_ensemble_network(input_shape, self.args)
        self.network6 = agent_ensemble_network(input_shape, self.args)
        self.network7 = agent_ensemble_network(input_shape, self.args)
        self.network8 = agent_ensemble_network(input_shape, self.args)
        self.network9 = agent_ensemble_network(input_shape, self.args)
        self.network10 = agent_ensemble_network(input_shape, self.args)    

    def init_hidden(self):
        # make hidden states on same device as model
        h1 = self.network1.init_hidden()
        h2 = self.network2.init_hidden()
        h3 = self.network3.init_hidden()
        h4 = self.network4.init_hidden()
        h5 = self.network5.init_hidden()
        h6 = self.network6.init_hidden()
        h7 = self.network7.init_hidden()
        h8 = self.network8.init_hidden()
        h9 = self.network9.init_hidden()
        h10 = self.network10.init_hidden()
        return [h1,h2,h3,h4,h5,h6,h7,h8,h9,h10]

    def forward(self, inputs, hidden_state):
        [h1,h2,h3,h4,h5,h6,h7,h8,h9,h10] = hidden_state
        q,h_1 = self.network1(inputs,h1)
        q2,h_2 = self.network2(inputs,h2)
        q3,h_3 = self.network3(inputs,h3)
        q4,h_4 = self.network4(inputs,h4)
        q5,h_5 = self.network5(inputs,h5)
        q6,h_6 = self.network6(inputs,h6)
        q7,h_7 = self.network7(inputs,h7)
        q8,h_8 = self.network8(inputs,h8)
        q9,h_9 = self.network9(inputs,h9)
        q10,h_10 = self.network10(inputs,h10)  

        q_total = (q+q2+q3+q4+q5+q6+q7+q8+q9+q10)/th.tensor(10)

        return q_total, [h_1,h_2,h_3,h_4,h_5,h_6,h_7,h_8,h_9,h_10]
'''
class RNNAgentN(nn.Module):
    def __init__(self, input_shape, args):
        super(RNNAgentN, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        return q, h
'''
