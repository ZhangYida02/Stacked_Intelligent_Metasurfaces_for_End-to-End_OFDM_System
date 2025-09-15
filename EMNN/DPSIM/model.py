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


class EMNN_tx(nn.Module):
    def __init__(self, Number_DPEM, L,txDpSim):
        super(EMNN_tx, self).__init__()
        self.L = L
        self.WTxDpA = torch.from_numpy(txDpSim['WTxDpA']).cuda()
        self.WTxDpSim = torch.from_numpy(txDpSim['WTxDpSim']).cuda()
        self.DPRIS = nn.ModuleList([nn.Embedding(1, 2 * Number_DPEM) for _ in range(self.L)])

    def forward(self, x, args):
        m, n = x.shape
        x_plural=x[:, :n//2]+1j*x[:, n//2:]
        x_plural = torch.transpose(x_plural, 1, 0).cfloat()
        TxDpSim_transmit_complex = torch.from_numpy(np.zeros((2 * args['M_dp'] * args['N_c'], m), dtype=np.complex64)).cuda()
        for n_c in range(args['N_c']):
            temp = self.WTxDpA[:, :, n_c] @ x_plural[n_c * 2 * args['A_tx_dp']: (n_c + 1) * 2 * args['A_tx_dp'], :]
            for i in range(self.L):
                phase=self.DPRIS[i](torch.tensor(0).cuda())
                DPRis_coefficient = torch.diag(torch.exp(1j*phase)).cfloat()
                temp = DPRis_coefficient @ temp
                if i != self.L-1:
                    temp = self.WTxDpSim[:,:,i,n_c] @ temp
            TxDpSim_transmit_complex[n_c * 2 * args['M_dp']: (n_c + 1) * 2 * args['M_dp'], :] = temp
        TxSim_transmit = torch.transpose(TxDpSim_transmit_complex, 1, 0)
        return TxSim_transmit


class EMNN_rx(nn.Module):
    def __init__(self, Number_DPEM, K, rxDpSim):
        super(EMNN_rx, self).__init__()
        self.K = K
        self.WRxA = torch.from_numpy(rxDpSim['WRxDpA']).cuda()
        self.WRxDpSim = torch.from_numpy(rxDpSim['WRxDpSim']).cuda()
        self.DPRIS = nn.ModuleList([nn.Embedding(1, 2 * Number_DPEM) for _ in range(self.K)])

    def forward(self, x0, args):
        m, n = x0.shape
        x_plural= x0[:, :n//2] + 1j * x0[:, :n//2]
        x_plural = torch.transpose(x_plural, 1, 0).cfloat()
        RxSim_receive_complex = torch.from_numpy(np.zeros((2 * args['A_rx_dp'] * args['N_c'], m), dtype=np.complex64)).cuda()
        for n_c in range(args['N_c']):
            temp = x_plural[n_c * 2 * args['N_dp']: (n_c + 1) * 2 * args['N_dp'], :]
            for i in range(self.K - 1, -1, -1):
                phase=self.DPRIS[i](torch.tensor(0).cuda())
                DPRis_coefficient = torch.diag(torch.exp(1j*phase)).cfloat()
                temp = DPRis_coefficient @ temp
                if i != 0:
                    temp = self.WRxDpSim[:,:,i-1,n_c] @ temp
            temp = self.WRxA[:, :, n_c] @ temp
            RxSim_receive_complex[n_c * 2 * args['A_rx_dp']: (n_c + 1) * 2 * args['A_rx_dp'], :] = temp
        RxSim_receive = torch.cat([RxSim_receive_complex.real, RxSim_receive_complex.imag], dim=0)
        RxSim_receive = torch.transpose(RxSim_receive, 1, 0)
        return RxSim_receive


def get_model(args,txDpSim,rxDpSim):
    NN_tx = DNN_tx(input_size=args['K_bit'], hidden_size=args['K_bit'] * args['N_c'], output_size=4 * args['A_tx_dp'] * args['N_c'])
    DPSIM_tx = EMNN_tx(args['M_dp'], args['L'],txDpSim)
    DPSIM_Rx = []
    NN_RX = []
    for j in range(args['J']):
        DPSIM_Rx.append(EMNN_rx(args['N_dp'], args['K'], rxDpSim))
        NN_RX.append(DNN_rx(input_size=4 * args['A_rx_dp'] * args['N_c'], hidden_size=args['v_bit'][j] * args['N_c'], output_size=args['v_bit'][j]))

    print(NN_tx)
    print(DPSIM_tx)
    print(DPSIM_Rx)
    print(NN_RX)
    return NN_tx, DPSIM_tx, DPSIM_Rx, NN_RX


