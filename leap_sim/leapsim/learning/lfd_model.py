"""
    This file contains the network definition for LfD
"""

import torch
from torch import nn


class LfDAgent(nn.Module):
    def __init__(self, input_size, hidden_size=256, rnn_layers=1, mlp_hidden_sizes=[512, 256, 128], output_size=16):
        super(LfDAgent, self).__init__()
        
        # RNN层
        self.rnn = nn.RNN(input_size, hidden_size, num_layers=rnn_layers, batch_first=True)
        
        # MLP层
        layers = []
        in_features = hidden_size
        for out_features in mlp_hidden_sizes:
            layers.append(nn.Linear(in_features, out_features))
            layers.append(nn.ReLU())
            in_features = out_features
        layers.append(nn.Linear(in_features, output_size))
        
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        # RNN层的前向传播
        rnn_out, _ = self.rnn(x)
        
        # 仅使用RNN输出的最后一个时间步作为MLP的输入
        last_rnn_output = rnn_out[:, -1, :]
        
        # MLP层的前向传播
        output = self.mlp(last_rnn_output)
        return output
    
    