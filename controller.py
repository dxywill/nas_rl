import numpy as np
import torch
import torch.nn as nn


class Agent(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_steps=4, device=''):
        super(Agent, self).__init__()

        # Could add an embedding layer
        # embedding_size = 100
        # self.embedding = nn.Embedding(input_size, embedding_size)
        # dropout layer
        #self.drop = nn.Dropout(dropout)
        self.DEVICE = device
        self.num_filter_option = 3
        self.filter_size_option = 3

        self.lstm1 = nn.LSTMCell(input_size, hidden_size)
        # May be could just use different decoder if these two numbers are the same, not sure
        self.decoder = nn.Linear(hidden_size, self.num_filter_option)
        #self.decoder2 = nn.Linear(hidden_size, self.filter_size_option)

        # num_steps = max_layer * 2 # two conv layer * 2 h-parameters (kernel size and number of kernels)
        self.num_steps = num_steps
        self.nhid = hidden_size
        self.hidden = self.init_hidden()

    def forward(self, input):
        outputs = []
        h_t, c_t = self.hidden

        for i in range(self.num_steps):
            # input_data = self.embedding(step_data)
            h_t, c_t = self.lstm1(input, (h_t, c_t))
            # Add drop out
            # h_t = self.drop(h_t)
            output = self.decoder(h_t)
            input = output
            outputs += [output]

        outputs = torch.stack(outputs).squeeze(1)

        return outputs

    def init_hidden(self):
        h_t = torch.zeros(1, self.nhid, dtype=torch.float, device=self.DEVICE)
        c_t = torch.zeros(1, self.nhid, dtype=torch.float, device=self.DEVICE)

        return (h_t, c_t)