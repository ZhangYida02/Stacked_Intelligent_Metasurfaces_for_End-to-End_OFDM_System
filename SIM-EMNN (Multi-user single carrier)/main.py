import random
import time
import numpy as np
from matplotlib import pyplot as plt
import figure
import model
import parameter
import system
import autoencoder
import channel
import os
from train import train
from test import test


os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


args=parameter.get_args()


random.seed(args['seed'])
np.random.seed(args['seed'])


txSim=system.get_txSim(args)
rxSim=system.get_rxSim(args)


time_start = time.time()


G = channel.get_channel(args, txSim, rxSim)


print('---------------------------------------------model---------------------------------------------')
NN_tx, SIM_tx, SIM_Rx, NN_RX = model.get_model(args, txSim, rxSim)
print(NN_tx)
print(SIM_tx)
print(SIM_Rx)
print(NN_RX)


print('---------------------------------------------train---------------------------------------------')
X_train, Y_train = autoencoder.generate_transmit_data(2 ** args['K_bit'], args['J'], args['Num_train'], seed=random.randint(0, 1000))
NN_tx, SIM_tx, SIM_Rx, NN_RX = train(X_train, Y_train, args, G ,NN_tx, SIM_tx, SIM_Rx, NN_RX)


print('---------------------------------------------rate---------------------------------------------')
X_rate, Y_rate = autoencoder.generate_rate_data(2 ** args['K_bit'], args['J'])
_, y_rate = test(X_rate, args, args['Noise_rate_dBm'], NN_tx, SIM_tx, SIM_Rx, NN_RX, G)
for j in range(args['J']):
    mean_rate, max_rate = autoencoder.calcul_rate(y_rate[j], args['Noise_rate_dBm'], args['A_rx'])
    print('The %d UE mean rate and max rate are %0.2fbps/Hz, %0.2fbps/Hz' % (j, mean_rate, max_rate))

print('---------------------------------------------test---------------------------------------------')
ber = np.zeros(args['Noise_test_dBm'].shape)

for i_noise_test in range(args['Noise_test_dBm'].shape[0]):
    X_test, Y_test = autoencoder.generate_transmit_data(2 ** args['K_bit'], args['J'], args['Num_test'], seed=random.randint(0, 1000))
    Y_pred, y_receiver = test(X_test, args, args['Noise_test_dBm'][i_noise_test], NN_tx, SIM_tx, SIM_Rx, NN_RX, G)
    ber[i_noise_test] = autoencoder.BER(X_test, Y_pred, args)
    print('The BER at noise_test_dbm=%d is %0.8f%%' % (args['Noise_test_dBm'][i_noise_test], 100 * ber[i_noise_test]))


print('---------------------------------------------plot---------------------------------------------')
figure.plot_ber(args['Noise_test_dBm'],ber)

time_end = time.time()
print(f'totally cost {time_end - time_start:.2f} s')

plt.show()