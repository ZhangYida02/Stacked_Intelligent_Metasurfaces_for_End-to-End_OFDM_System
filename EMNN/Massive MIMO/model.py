import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class DNN_tx(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DNN_tx, self).__init__()

        # self.precoder = nn.Embedding(1, output_size)
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, output_size)

        torch.nn.init.kaiming_normal_(self.map1.weight, a=0, mode='fan_in', nonlinearity='relu')
        torch.nn.init.constant_(self.map1.bias, 1)
        torch.nn.init.kaiming_normal_(self.map2.weight, a=0, mode='fan_in', nonlinearity='relu')
        torch.nn.init.constant_(self.map2.bias, 1)

    def forward(self, x):
        x = F.relu(self.map1(x))
        x = F.relu(self.map2(x))
        return self.map3(x)


class DNN_rx(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DNN_rx, self).__init__()
        self.bn1 = nn.BatchNorm1d(input_size)
        self.map1 = nn.Linear(input_size, hidden_size, bias=True)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.map2 = nn.Linear(hidden_size, output_size, bias=True)
        self.bn3 = nn.BatchNorm1d(output_size)

        torch.nn.init.kaiming_normal_(self.map1.weight, a=0, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(self.bn1.weight, 1)
        torch.nn.init.kaiming_normal_(self.map2.weight, a=0, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(self.bn2.weight, 1)

    def forward(self, x):
        x = self.bn1(x)
        x = F.relu(self.map1(x))
        x = self.bn2(x)
        x = F.relu(self.map2(x))
        x = self.bn3(x)
        return torch.sigmoid(x)


def get_model(args):
    NN_tx = DNN_tx(input_size=args['K_bit'], hidden_size=args['K_bit'] * args['N_c'], output_size=2 * args['A_tx'] * args['N_c'])
    NN_RX = []
    for j in range(args['J']):
        NN_RX.append(DNN_rx(input_size=2 * args['A_rx'] * args['N_c'], hidden_size=args['v_bit'][j] * args['N_c'], output_size=args['v_bit'][j]))

    print(NN_tx)
    print(NN_RX)
    return NN_tx, NN_RX


