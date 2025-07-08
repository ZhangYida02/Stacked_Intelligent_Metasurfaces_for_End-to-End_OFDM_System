import torch
import numpy as np
import autoencoder
from channel import Channel

def test(X_test, args, noise_test_dbm, NN_tx, SIM_tx, SIM_RX, NN_RX, G):
    num_test = X_test.shape[0]

    Noise_test = []
    noise_test_dbm = 10 ** (noise_test_dbm / 10) * 10 ** -3
    for J in range(args['J']):
        Noise_test.append(torch.from_numpy(np.random.randn(num_test , args['A_rx'] * 2) * np.sqrt(noise_test_dbm / 2)).float().cuda())

    x_transmit = NN_tx(torch.from_numpy(autoencoder.onehot2bit(X_test, args['J'], args['K_bit'])).float())
    x_transmit = x_transmit.cuda()

    norm = torch.empty(1, num_test )
    norm[0, :] = torch.norm(x_transmit, 2, 1)
    norm = norm.cuda()

    x_transmit = x_transmit / torch.t(norm)

    TxSim_transmit = SIM_tx(x_transmit)

    RxSim_receive = Channel(TxSim_transmit, G, args['J'])


    y_pred = []
    Y_receive = []
    for j in range(args['J']):
        y_receive = SIM_RX[j](RxSim_receive[j]) + Noise_test[j]
        y_pred.append(NN_RX[j](y_receive.float()))
        y_receive = y_receive.detach().cpu().numpy()
        Y_receive.append(y_receive)

    y_pred = torch.reshape(torch.stack(y_pred, 1), [num_test, args['K_bit'] * args['J']])
    y_pred = y_pred.detach().cpu().numpy()

    return y_pred, Y_receive