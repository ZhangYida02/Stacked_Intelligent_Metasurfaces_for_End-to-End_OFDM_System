import argparse
import math
import numpy as np
import torch
import random

def get_args():
    parser = argparse.ArgumentParser(description="hyperparameters")
    parser.add_argument('--cuda', default=torch.cuda.is_available(), type=bool, help="Device to run on ('cpu' or 'cuda').")
    parser.add_argument('--seed', default=28, type=int, help="Random seed for reproducibility.")
    parser.add_argument('--Monte_Carlo', default='100', type=int, help="Monte Carlo.")
    parser.add_argument('--algo_name', default='EMNN', type=str, help="Name of the algorithm.")
    parser.add_argument('--env_name', default='E2E', type=str, help="Name of the environment.")
    parser.add_argument('--batch_size', default=1000, type=int, help="Batch size for training updates.")
    parser.add_argument('--batch_size_test', default=20000, type=int, help="Batch size for training updates.")
    parser.add_argument('--train_eps', default=2000, type=int, help="Number of training episodes.")
    parser.add_argument('--Num_vali', default=2000, type=int, help="Single-round verification scale.")
    parser.add_argument('--Num_test', default=100000, type=int, help="Single-round test scale.")
    parser.add_argument('--A_tx', default='256', type=int, help="Number of transmitting antennas.")
    parser.add_argument('--A_rx', default='49', type=int, help="Number of receiving antennas.")
    parser.add_argument('--J', default='3', type=int, help="Number of UE.")
    parser.add_argument('--Noise_dBm', default='-110', type=int, help="Noise (dBm).")
    parser.add_argument('--F_c', default='28', type=int, help="Frequency (GHz).")
    parser.add_argument('--B', default='100', type=int, help="Bandwidth (MHz).")
    parser.add_argument('--epsilon', default='0.2', type=float, help="Polarization channel energy conversion")
    parser.add_argument('--N_c', default='32', type=int, help="number of subcarriers.")
    parser.add_argument('--Plrd', default='1', type=float, help="Path loss reference distance (m).")
    parser.add_argument('--Ple', default='3.5', type=float, help="Path loss exponent.")
    parser.add_argument('--Plsfv_dB', default='9', type=float, help="Path loss shadowing fading variance (dB).")
    parser.add_argument('--LR_Factor', default='1.05', type=float, help="Learning rate decay.")

    args = parser.parse_args()
    args = {**vars(args)}

    print(''.join(['=']*80))
    tplt = "{:^20}\t{:^20}\t{:^20}"
    print(tplt.format("参数名","参数值","参数类型"))
    for k,v in args.items():
        print(tplt.format(k,v,str(type(v))))
    print(''.join(['=']*80))

    #args['channel'] = 'statistics.mat'
    args['channel'] = '55.mat'

    if args['channel'] == 'statistics.mat':
        args['P_train_dBm'] = [25, 100]  # 统计 [25,100]
    else :
        args['P_train_dBm'] = [5, 85]  # 直接瞬时 [0,85]

    args['Lam_c'] = 3*10**8/(args['F_c']*10**9)
    args['A_tx1'] = int(math.sqrt(args['A_tx']))
    args['A_tx2'] = int(math.sqrt(args['A_tx']))
    args['A_rx1'] = int(math.sqrt(args['A_rx']))
    args['A_rx2'] = int(math.sqrt(args['A_rx']))
    args['dxTx'] = args['Lam_c']/2
    args['dyTx'] = args['Lam_c']/2
    args['dxRx'] = args['Lam_c']/2
    args['dyRx'] = args['Lam_c']/2
    args['v_bit'] = [8,8,8] #32，16，8
    args['Noise'] = 10 ** (args['Noise_dBm'] / 10) * 10 ** -3
    args['P_tx_vali_dbm'] = np.arange(-20, 30, 5)
    args['P_tx_test_dBm'] = np.arange(-20, 30, 1)
    args['F'] = np.linspace(args['F_c'] - args['B'] * 10 ** -3 / 2, args['F_c'] + args['B'] * 10 ** -3 / 2, args['N_c'])
    args['Lam'] = 3 * 10 ** 8 / (args['F'] * 10 ** 9)
    args['K_bit']=sum(args['v_bit'])



    folder_name = f"Unpolarized channel/{args['B']}MHz_{args['N_c']}Nc_{args['J']}UE_{args['epsilon']}ep_{args['A_tx']}-{args['A_rx']}"
    file_name = args['channel']
    args['channel_path'] = folder_name+"/"+file_name

    name = f"{args['B']}MHz_{args['N_c']}Nc_{args['J']}UE_{args['epsilon']}ep_{args['A_tx']}-{args['A_rx']}"
    args['pathname_prefix'] =  name

    random.seed(args['seed'])
    np.random.seed(args['seed'])


    return args