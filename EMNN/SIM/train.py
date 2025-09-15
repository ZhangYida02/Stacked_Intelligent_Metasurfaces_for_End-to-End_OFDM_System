import os
import random

import numpy as np
from matplotlib import pyplot as plt

from test import test
import torch
import autoencoder
from channel import Channel

optim_betas = (0.9, 0.999)
t_learning_rate = 0.005
r_learning_rate = 0.005
weight_decay = 0.0001
criterion = torch.nn.BCELoss()


def train(args, G, NN_tx, SIM_tx, SIM_RX, NN_RX):


    NN_tx = torch.nn.DataParallel(NN_tx).cuda()
    SIM_tx = torch.nn.DataParallel(SIM_tx).cuda()
    for j in range(args['J']):
        SIM_RX[j] = torch.nn.DataParallel(SIM_RX[j]).cuda()
        NN_RX[j] = torch.nn.DataParallel(NN_RX[j]).cuda()

    NN_tx_optimizer = torch.optim.AdamW(NN_tx.parameters(), lr=t_learning_rate, betas=optim_betas)
    SIM_tx_optimizer = torch.optim.AdamW(SIM_tx.parameters(), lr=t_learning_rate, betas=optim_betas)
    SIM_RX_optimizer = []
    NN_RX_optimizer = []
    for i in range(args['J']):
        SIM_RX_optimizer.append(torch.optim.AdamW(SIM_RX[i].parameters(), lr=t_learning_rate, betas=optim_betas))
        NN_RX_optimizer.append(torch.optim.AdamW(NN_RX[i].parameters(), lr=r_learning_rate, betas=optim_betas))

    R_error = []

    for epoch in range(args['train_eps']):

        low, high = args['P_train_dBm']


        #Beta分布 [3 1 1]
        progress = epoch / (args['train_eps'] - 1)
        bias = 3 * (progress - 0.5)
        a = 1 + max(bias, 0) * 1
        b_param = 1 + max(-bias, 0) * 1
        r = np.random.beta(a, b_param)
        P_train_dBm = int(low + r * (high - low))

        #P_train_dBm=70

        print(' epoch=',epoch,', P_train_dBm=',P_train_dBm)

        P_train = 10 ** (P_train_dBm / 10) * 10 ** -3

        X_train, Y_train = autoencoder.generate_transmit_data(2 ** args['K_bit'], args['batch_size'])

        Noise_train = []
        for j in range(args['J']):
            Noise_train.append(torch.from_numpy(np.random.randn(args['batch_size'], args['A_rx'] * args['N_c'] * 2) * np.sqrt(args['Noise']) / np.sqrt(2)).float().cuda())

        NN_tx.zero_grad()
        SIM_tx.zero_grad()
        for j in range(args['J']):
            SIM_RX[j].zero_grad()
            NN_RX[j].zero_grad()

        x_transmit = NN_tx(torch.from_numpy(autoencoder.onehot2bit(X_train, args['K_bit'])).float()).cuda()

        norm = torch.empty(args['batch_size'],1).cuda()
        norm[:,0] = torch.norm(x_transmit, 2, 1)
        x_transmit = x_transmit / norm * np.sqrt(P_train)

        target = torch.from_numpy(autoencoder.onehot2bit(Y_train, args['K_bit'])).float().cuda()

        TxSim_transmit = SIM_tx(x_transmit, args)

        RxSim_receive = Channel(TxSim_transmit, G, args['J'], args)

        r_error = 0
        for j in range(args['J']):
            RxSim_receive0 = torch.cat([RxSim_receive[j].real, RxSim_receive[j].imag], dim=1)
            y_receive = SIM_RX[j](RxSim_receive0, args) + Noise_train[j]
            r_error = r_error + criterion(NN_RX[j](y_receive.float()),target[:, sum(args['v_bit'][0:j]): sum(args['v_bit'][0:j + 1])])

        r_error.backward(retain_graph=True)

        for j in range(args['J']):
            NN_RX_optimizer[j].step()
            SIM_RX_optimizer[j].step()
        SIM_tx_optimizer.step()
        NN_tx_optimizer.step()

        R_error.append(r_error.cpu().detach().numpy())

        if epoch % 100 == 0:
            print("Epoch: %s, Loss: %s, LR: %0.5f" % (epoch, r_error, NN_tx_optimizer.param_groups[0]['lr']))

            NN_tx_optimizer.param_groups[0]['lr'] /= args['LR_Factor']
            SIM_tx_optimizer.param_groups[0]['lr'] /= args['LR_Factor']
            for i in range(args['J']):
                SIM_RX_optimizer[j].param_groups[0]['lr'] /= args['LR_Factor']
                NN_RX_optimizer[j].param_groups[0]['lr'] /= args['LR_Factor']

            ber = np.zeros(args['P_tx_vali_dbm'].shape)
            for i_P_tx in range(args['P_tx_vali_dbm'].shape[0]):
                X_vali, Y_vali = autoencoder.generate_transmit_data(2 ** args['K_bit'], args['Num_vali'])
                y_pred = test(X_vali, args, args['P_tx_vali_dbm'][i_P_tx], NN_tx, SIM_tx, SIM_RX, NN_RX, G)
                ber[i_P_tx] = autoencoder.BER(Y_vali, y_pred, args)
                print('The BER at P_tx_vali_dbm=%d is %0.6f%%' % (args['P_tx_vali_dbm'][i_P_tx], 100 * ber[i_P_tx]))




    return NN_tx, SIM_tx, SIM_RX, NN_RX,R_error