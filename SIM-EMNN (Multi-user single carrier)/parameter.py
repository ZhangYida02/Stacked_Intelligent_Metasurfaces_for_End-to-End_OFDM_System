import argparse
import math
import numpy as np
import torch

def get_args():
    parser = argparse.ArgumentParser(description="hyperparameters")
    parser.add_argument('--cuda', default=torch.cuda.is_available(), type=bool, help="Device to run on ('cpu' or 'cuda').")
    parser.add_argument('--seed', default=1, type=int, help="Random seed for reproducibility.")
    parser.add_argument('--Monte_Carlo', default='10', type=int, help="Monte Carlo.")
    parser.add_argument('--algo_name', default='EMNN', type=str, help="Name of the algorithm.")
    parser.add_argument('--env_name', default='E2E-SIM', type=str, help="Name of the environment.")
    parser.add_argument('--batch_size', default=200, type=int, help="Batch size for training updates.")
    parser.add_argument('--train_eps', default=400, type=int, help="Number of training episodes.")
    parser.add_argument('--Num_train', default=10000, type=int, help="Single-round training scale.")
    parser.add_argument('--Num_vali', default=10000, type=int, help="Single-round verification scale.")
    parser.add_argument('--Num_test', default=5 * 10 ** 5, type=int, help="Single-round test scale.")
    parser.add_argument('--L', default='3', type=int, help="Number of layers of the TX-SIM metamaterial layer.")
    parser.add_argument('--K_bit', default='4', type=int, help="Modulation bit number.")
    parser.add_argument('--A_tx', default='16', type=int, help="Number of transmitting antennas.")
    parser.add_argument('--A_rx', default='4', type=int, help="Number of receiving antennas.")
    parser.add_argument('--K', default='3', type=int, help="Number of layers of the RX-SIM metamaterial layer.")
    parser.add_argument('--J', default='4', type=int, help="Number of UE.")
    parser.add_argument('--M', default='100', type=int, help="Number of EM units per layer of TX-SIM.")
    parser.add_argument('--N', default='100', type=int, help="Number of EM units per layer of RX-SIM.")
    parser.add_argument('--D_tx', default='0.05', type=float, help="TX-SIM thickness (m).")
    parser.add_argument('--D_rx', default='0.05', type=float, help="RX-SIM thickness (m).")
    parser.add_argument('--P_dBm', default='10', type=int, help="Transmit power (dBm).")
    parser.add_argument('--Noise_train_dBm', default='-130', type=int, help="Train noise (dBm).")
    parser.add_argument('--Noise_rate_dBm', default='-115', type=int, help="Test rate noise (dBm).")
    parser.add_argument('--F', default='28', type=int, help="Frequency (GHz).")
    parser.add_argument('--Plrd', default='1', type=float, help="Path loss reference distance (m).")
    parser.add_argument('--Ple', default='3.5', type=float, help="Path loss exponent.")
    parser.add_argument('--Plsfv_dB', default='9', type=float, help="Path loss shadowing fading variance (dB).")
    parser.add_argument('--LR_Factor', default='1.2', type=float, help="Learning rate decay.")

    args = parser.parse_args()
    args = {**vars(args)}

    print(''.join(['=']*80))
    tplt = "{:^20}\t{:^20}\t{:^20}"
    print(tplt.format("参数名","参数值","参数类型"))
    for k,v in args.items():
        print(tplt.format(k,v,str(type(v))))
    print(''.join(['=']*80))

    args['Lam'] = 3*10**8/(args['F']*10**9)
    args['M1'] = int(math.sqrt(args['M']))
    args['M2'] = int(math.sqrt(args['M']))
    args['A_tx1'] = int(math.sqrt(args['A_tx']))
    args['A_tx2'] = int(math.sqrt(args['A_tx']))
    args['N1'] = int(math.sqrt(args['N']))
    args['N2'] = int(math.sqrt(args['N']))
    args['A_rx1'] = int(math.sqrt(args['A_rx']))
    args['A_rx2'] = int(math.sqrt(args['A_rx']))
    args['dzTx'] = args['D_tx']/(args['L'])
    args['dxTx'] = args['Lam']/2
    args['dyTx'] = args['Lam']/2
    args['dzRx'] = args['D_rx']/(args['K'])
    args['dxRx'] = args['Lam']/2
    args['dyRx'] = args['Lam']/2
    args['Noise_train'] = 10 ** (args['Noise_train_dBm'] / 10) * 10 ** -3
    args['Noise_test_dBm'] = np.arange(-120, -90, 2)
    args['D_tx_rx']=[60, 70, 80, 90, 100]


    return args