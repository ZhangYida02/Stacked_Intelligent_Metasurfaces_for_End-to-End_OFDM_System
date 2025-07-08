import os
import random
import numpy as np
from matplotlib import pyplot as plt
import model
from test import test
import torch
import autoencoder
from channel import Channel


optim_betas = (0.9, 0.999)
t_learning_rate = 0.003
r_learning_rate = 0.003
weight_decay = 0.0001
criterion = torch.nn.BCELoss()


def train(X_train, Y_train, args, G ,NN_tx, SIM_tx, SIM_RX, NN_RX):
    gpu_list = '0'
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list

    total_batch = int(args['Num_train'] / args['batch_size'])

    if (args['cuda']):
        NN_tx = torch.nn.DataParallel(NN_tx).cuda()
        SIM_tx = torch.nn.DataParallel(SIM_tx).cuda()
        for i in range(args['J']):
            SIM_RX[i] = torch.nn.DataParallel(SIM_RX[i]).cuda()
            NN_RX[i] = torch.nn.DataParallel(NN_RX[i]).cuda()

    # NN_tx_optimizer = torch.optim.Adam(NN_tx.parameters(), lr=t_learning_rate, betas=optim_betas)
    # SIM_tx_optimizer = torch.optim.Adam(SIM_tx.parameters(), lr=t_learning_rate, betas=optim_betas)
    NN_tx_optimizer = torch.optim.AdamW(NN_tx.parameters(), lr=t_learning_rate, betas=optim_betas)
    SIM_tx_optimizer = torch.optim.AdamW(SIM_tx.parameters(), lr=t_learning_rate, betas=optim_betas)
    SIM_RX_optimizer = []
    NN_RX_optimizer = []
    for i in range(args['J']):
        SIM_RX_optimizer.append(torch.optim.AdamW(SIM_RX[i].parameters(), lr=t_learning_rate, betas=optim_betas))
        NN_RX_optimizer.append(torch.optim.AdamW(NN_RX[i].parameters(), lr=r_learning_rate, betas=optim_betas))


    R_error = []

    for epoch in range(args['train_eps']):
        error_epoch = 0
        for index in range(total_batch):
            Noise_train = []
            for J in range(args['J']):
                Noise_train.append(torch.from_numpy(np.random.randn(args['batch_size'], args['A_rx'] * 2) * np.sqrt(args['Noise_train']) / np.sqrt(2)).float().cuda())

            idx = np.random.randint(args['Num_train'], size=args['batch_size'])

            NN_tx.zero_grad()
            SIM_tx.zero_grad()
            for i in range(args['J']):
                SIM_RX[i].zero_grad()
                NN_RX[i].zero_grad()

            x_transmit = NN_tx(torch.from_numpy(autoencoder.onehot2bit(X_train[idx, :], args['J'], args['K_bit'])).float())
            x_transmit = x_transmit.cuda()

            target = torch.from_numpy(autoencoder.onehot2bit(Y_train[idx, :], args['J'], args['K_bit'])).float()
            target = target.cuda()

            norm = torch.empty(1, args['batch_size'])
            norm[0, :] = torch.norm(x_transmit, 2, 1)
            norm = norm.cuda()

            x_transmit = x_transmit / torch.t(norm)

            TxSim_transmit = SIM_tx(x_transmit)

            RxSim_receive = Channel(TxSim_transmit, G, args['J'])

            r_error = 0

            for j in range(args['J']):
                y_receive = SIM_RX[j](RxSim_receive[j]) + Noise_train[j]
                r_error = r_error + criterion(NN_RX[j](y_receive.float()), target[:, j * args['K_bit']:(j + 1) * args['K_bit']])

            r_error.backward(retain_graph=True)

            for j in range(args['J']):
                NN_RX_optimizer[j].step()
                SIM_RX_optimizer[j].step()
            SIM_tx_optimizer.step()
            NN_tx_optimizer.step()

            error_epoch = error_epoch + r_error
        R_error.append(error_epoch.cpu().detach().numpy())

        if epoch % 10 == 0:
            print("Epoch: %s, Loss: %s, LR: %0.5f" % (epoch, extract(error_epoch), NN_tx_optimizer.param_groups[0]['lr']))

            NN_tx_optimizer.param_groups[0]['lr'] /= args['LR_Factor']
            SIM_tx_optimizer.param_groups[0]['lr'] /= args['LR_Factor']
            for i in range(args['J']):
                SIM_RX_optimizer[j].param_groups[0]['lr'] /= args['LR_Factor']
                NN_RX_optimizer[j].param_groups[0]['lr'] /= args['LR_Factor']

            noise_vali_dbm = np.array([-140, -135,-130, -125, -120, -115, -110, -105])
            ber = np.zeros(noise_vali_dbm.shape)

            for i_noise in range(noise_vali_dbm.shape[0]):
                X_vali, Y_vali = autoencoder.generate_transmit_data(2**args['K_bit'], args['J'], args['Num_vali'], seed=random.randint(0, 1000))
                y_pred, y_receive = test(X_vali, args, noise_vali_dbm[i_noise], NN_tx, SIM_tx, SIM_RX, NN_RX, G)
                ber[i_noise] = autoencoder.BER(X_vali, y_pred, args)
                print('The BER at noise_vali_dbm=%d is %0.3f%%' % (noise_vali_dbm[i_noise], 100 * ber[i_noise]))

    plt.figure(1)
    plt.plot(R_error)

    return NN_tx, SIM_tx, SIM_RX, NN_RX




















def extract(v):
    return v.data.storage().tolist()







class Regularization(torch.nn.Module):
    def __init__(self, model, weight_decay, p=2):
        super(Regularization, self).__init__()
        if weight_decay <= 0:
            print("param weight_decay can not <=0")
            exit(0)
        self.model = model
        self.weight_decay = weight_decay
        self.p = p
        self.weight_list = self.get_weight(model)
        self.weight_info(self.weight_list)

    def to(self, device):
        self.device = device
        super().to(device)
        return self

    def forward(self, model):
        self.weight_list = self.get_weight(model)
        reg_loss = self.regularization_loss(self.weight_list, self.weight_decay, p=self.p)
        return reg_loss

    def get_weight(self, model):
        weight_list = []
        for name, param in model.named_parameters():
            if 'weight' in name:
                weight = (name, param)
                weight_list.append(weight)
        return weight_list

    def regularization_loss(self, weight_list, weight_decay, p=2):
        reg_loss = 0
        for name, w in weight_list:
            l2_reg = torch.norm(w, p=p)
            reg_loss = reg_loss + l2_reg

        reg_loss = weight_decay * reg_loss
        return reg_loss

    def weight_info(self, weight_list):
        pass

