import torch
from torch import nn
import torch.nn.functional as F


weight_gain = 1
bias_gain = 0.1

class DNN_tx(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DNN_tx, self).__init__()

        # self.precoder = nn.Embedding(1, output_size)
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, output_size)

        torch.nn.init.xavier_normal_(self.map1.weight, weight_gain)
        torch.nn.init.constant(self.map1.bias, bias_gain)
        torch.nn.init.xavier_normal_(self.map2.weight, weight_gain)
        torch.nn.init.constant(self.map2.bias, bias_gain)

    def forward(self, x):
        x = F.relu(self.map1(x))
        x = F.relu(self.map2(x))
        return self.map3(x)

class DNN_rx(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DNN_rx, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size, bias=True)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.map2 = nn.Linear(hidden_size, output_size, bias=True)
        self.bn2 = nn.BatchNorm1d(output_size)

    def forward(self, x):
        x = F.relu(self.map1(x))
        x = self.bn1(x)
        x = (self.map2(x))
        x = self.bn2(x)
        return torch.sigmoid(x)


class EMNN_tx(nn.Module):
    def __init__(self, Number_EM, L,txSim):
        super(EMNN_tx, self).__init__()
        self.L = L
        self.WTxA = torch.from_numpy(txSim['WTxA']).cuda()
        self.WTxSim = torch.from_numpy(txSim['WTxSim']).cuda()
        self.RIS = nn.ModuleList([nn.Embedding(1, Number_EM) for _ in range(self.L)])

    def forward(self, x):
        m, n = x.shape
        x_plural=x[:, :n//2]+1j*x[:, n//2:]
        x_plural = torch.transpose(x_plural, 1, 0).cdouble()
        temp = self.WTxA @ x_plural
        for i in range(self.L):
            phase=self.RIS[i](torch.tensor(0).cuda())
            Ris_coefficient = torch.diag(torch.exp(1j*phase)).cdouble()
            temp = Ris_coefficient @ temp
            if i != self.L-1:
                temp = self.WTxSim[:,:,i] @ temp

        TxSim_transmit = torch.cat([temp.real, temp.imag], dim=0)
        TxSim_transmit = torch.transpose(TxSim_transmit, 1, 0)
        return TxSim_transmit


class EMNN_rx(nn.Module):
    def __init__(self, Number_EM, K,rxSim):
        super(EMNN_rx, self).__init__()
        self.K = K
        self.WRxA = torch.from_numpy(rxSim['WRxA']).cuda()
        self.WRxSim = torch.from_numpy(rxSim['WRxSim']).cuda()
        self.RIS = nn.ModuleList([nn.Embedding(1, Number_EM) for _ in range(self.K)])

    def forward(self, x):
        m, n = x.shape
        x_plural=x[:, :n//2]+1j*x[:, n//2:]
        x_plural = torch.transpose(x_plural, 1, 0).cdouble()
        temp = x_plural
        for i in range(self.K - 1, -1, -1):
            phase=self.RIS[i](torch.tensor(0).cuda())
            Ris_coefficient = torch.diag(torch.exp(1j*phase)).cdouble()
            temp = Ris_coefficient @ temp
            if i != 0:
                temp = self.WRxSim[:,:,i-1] @ temp

        temp = self.WRxA @ temp

        RxSim_receive = torch.cat([temp.real, temp.imag], dim=0)
        RxSim_receive = torch.transpose(RxSim_receive, 1, 0)

        return RxSim_receive


def get_model(args,txSim,rxSim):
    NN_tx = DNN_tx(input_size=args['K_bit'] * args['J'], hidden_size=4 * 2 ** args['K_bit'] * args['J'], output_size=args['A_tx'] * 2)
    SIM_tx = EMNN_tx(args['M'], args['L'],txSim)
    SIM_Rx = []
    NN_RX = []
    for i in range(args['J']):
        SIM_Rx.append(EMNN_rx(args['N'], args['K'], rxSim))
        NN_RX.append(DNN_rx(input_size=args['A_rx'] * 2, hidden_size=4 * 2 ** args['K_bit'], output_size=args['K_bit']))
    return NN_tx, SIM_tx, SIM_Rx, NN_RX


