import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
gpu_list = '1'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list

import time
import numpy as np
from matplotlib import pyplot as plt
import figure
import model
import parameter
import system
import autoencoder
import channel


from save import save_all_models, load_all_models, save_results_mat
from train import train
from test import test

time_start = time.time()

print('---------------------------------------------parameter---------------------------------------------')
args=parameter.get_args()

print('---------------------------------------------SIM---------------------------------------------')
txSim=system.get_txSim(args)
rxSim=system.get_rxSim(args)

print('---------------------------------------------channel---------------------------------------------')
G = channel.get_channel(args, txSim, rxSim)

print('---------------------------------------------model---------------------------------------------')
NN_tx, SIM_tx, SIM_Rx, NN_RX = model.get_model(args, txSim, rxSim)

print('---------------------------------------------load---------------------------------------------')
if args['channel'] != 'statistics.mat':
    #NN_tx, SIM_tx, SIM_Rx, NN_RX = load_all_models(NN_tx, SIM_tx, SIM_Rx, NN_RX, args['pathname_prefix'], 'cuda')
    pass
else :
    print(f"不加载模型")

print('---------------------------------------------train---------------------------------------------')
NN_tx, SIM_tx, SIM_Rx, NN_RX,R_error= train(args, G ,NN_tx, SIM_tx, SIM_Rx, NN_RX)

print('---------------------------------------------test---------------------------------------------')
ber = np.zeros(args['P_tx_test_dBm'].shape)

for p_tx_test in range(args['P_tx_test_dBm'].shape[0]):
    X_test, Y_test = autoencoder.generate_transmit_data(2 ** args['K_bit'], args['Num_test'])

    Y_pred=np.zeros((args['Num_test'],args['K_bit']))
    for test_index in range(args['Num_test']//args['batch_size_test']):
        Y_pred[test_index * args['batch_size_test']: (test_index + 1) * args['batch_size_test'], :] = test(X_test[test_index * args['batch_size_test']: (test_index + 1) * args['batch_size_test'], :], args, args['P_tx_test_dBm'][p_tx_test], NN_tx, SIM_tx, SIM_Rx, NN_RX, G)

    ber[p_tx_test] = autoencoder.BER(Y_test, Y_pred, args)
    print('The BER at p_tx_test_dbm=%.1f is %0.8f%%' % (args['P_tx_test_dBm'][p_tx_test], 100 * ber[p_tx_test]))




print('---------------------------------------------plot---------------------------------------------')
figure.plot_ber(args['P_tx_test_dBm'],ber)
figure.plot_loss(R_error)

plt.show()

print('---------------------------------------------save---------------------------------------------')
if args['channel'] == 'statistics.mat':
    save_all_models(NN_tx, SIM_tx, SIM_Rx, NN_RX, args['pathname_prefix'])
else:
    print(f"不保存模型")

save_results_mat( args, R_error, ber, args['pathname_prefix'])

print('---------------------------------------------time---------------------------------------------')

time_end = time.time()
print(f'totally cost {time_end - time_start:.2f} s')