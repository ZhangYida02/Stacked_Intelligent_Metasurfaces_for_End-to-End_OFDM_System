import math
import numpy as np
import torch
import os
import scipy.io as sio
from pathlib import Path
from scipy.linalg import sqrtm

"""
def get_channel(args, txSim, rxSim):
    RTx = np.zeros((args['M1'] * args['M2'], args['M1'] * args['M2']))
    RRx = np.zeros((args['N1'] * args['N2'], args['N1'] * args['N2']))

    for m1 in range(args['M1'] * args['M2']):
        for m2 in range(args['M1'] * args['M2']):
            rowl1, coll1 = np.unravel_index(m1, (args['M1'], args['M2']))
            rowl2, coll2 = np.unravel_index(m2, (args['M1'], args['M2']))
            t_val = np.sqrt((txSim['XTxSim'][rowl1] - txSim['XTxSim'][rowl2]) ** 2 + (txSim['YTxSim'][coll1] - txSim['YTxSim'][coll2]) ** 2)
            RTx[m1, m2] = np.sinc(2 * t_val / args['Lam_c'])

    for n1 in range(args['N1'] * args['N2']):
        for n2 in range(args['N1'] * args['N2']):
            rowl1, coll1 = np.unravel_index(n1, (args['N1'], args['N2']))
            rowl2, coll2 = np.unravel_index(n2, (args['N1'], args['N2']))
            r_val = np.sqrt((rxSim['XRxSim'][rowl1] - rxSim['XRxSim'][rowl2]) ** 2 + (rxSim['YRxSim'][coll1] - rxSim['YRxSim'][coll2]) ** 2)
            RRx[n1, n2] = np.sinc(2 * r_val / args['Lam_c'])

    G=[]
    for j in range(args['J']):
        G_Cor = np.zeros((args['N'], args['M'], args['N_c']), dtype=np.complex64)
        random_matrix = (np.random.randn(args['N'], args['M'], args['N_c']) + 1j * np.random.randn(args['N'], args['M'], args['N_c'])) / np.sqrt(2)
        for n_c in range(args['N_c']):
            PL = 20 * math.log10(4 * math.pi / args['Lam'][n_c]) + 10 * args['Ple'] * math.log10(args['D_tx_rx'][j]) + args['Plsfv_dB']
            rho_sq = 1 / 10 ** (PL / 10)
            GTilde = np.sqrt(rho_sq) * random_matrix[:,:,n_c]
            G_Cor[:,:,n_c]=sqrtm(RRx) @ GTilde @ sqrtm(RTx)
        G.append(G_Cor)
    return G
"""
def get_channel(args, txSim, rxSim):
    print(f"信道已从 {args['channel_path']} 加载")
    data = sio.loadmat(args['channel_path'])
    G= data['G']
    return G




def Channel(x_transmit, G, J, args):
    m,n=x_transmit.shape
    x_transmit = torch.transpose(x_transmit.cfloat(), 1, 0)
    Y_receive = []
    for j in range(J):
        y_receive = torch.from_numpy(np.zeros((args['N'] * args['N_c'], m), dtype=np.complex64)).cuda()
        for n_c in range(args['N_c']):
            G1=torch.from_numpy(G[:,:,:,j]).cuda().cfloat()
            y_receive_single=G1[:,:,n_c]@ x_transmit[n_c * args['M']: (n_c + 1) * args['M'], :]
            y_receive[n_c * args['N']: (n_c + 1) * args['N'], :]=y_receive_single
        Y_receive.append(torch.transpose(y_receive, 1, 0))
    return Y_receive
