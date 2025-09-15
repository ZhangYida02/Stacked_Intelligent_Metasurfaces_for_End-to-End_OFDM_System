import torch
import numpy as np
import autoencoder
from channel import Channel

def test(X_test, args, P_tx_dbm, NN_tx, SIM_tx, SIM_RX, NN_RX, G):
    num_test = X_test.shape[0]

    Noise_test = []
    for J in range(args['J']):
        Noise_test.append(torch.from_numpy(np.random.randn(num_test, args['A_rx'] * args['N_c'] * 2) * np.sqrt(args['Noise']) / np.sqrt(2)).float().cuda())

    P_tx_test = 10 ** (P_tx_dbm / 10) * 10 ** -3

    x_transmit = NN_tx(torch.from_numpy(autoencoder.onehot2bit(X_test, args['K_bit'])).float())


    norm = torch.empty(num_test, 1).cuda()
    norm[:, 0] = torch.norm(x_transmit, 2, 1)
    x_transmit = x_transmit / norm * np.sqrt(P_tx_test)

    TxSim_transmit = SIM_tx(x_transmit, args)

    RxSim_receive = Channel(TxSim_transmit, G, args['J'], args)

    y_pred = torch.zeros((num_test, args['K_bit']))
    for j in range(args['J']):
        RxSim_receive0 = torch.cat([RxSim_receive[j].real, RxSim_receive[j].imag], dim=1)
        y_receive = SIM_RX[j](RxSim_receive0, args) + Noise_test[j]
        y_pred[:,sum(args['v_bit'][0:j]): sum(args['v_bit'][0:j+1])]=NN_RX[j](y_receive.float())

    y_pred = y_pred.detach().cpu().numpy()

    return y_pred