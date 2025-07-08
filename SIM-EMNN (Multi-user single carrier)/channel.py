import math
import numpy as np
import torch
from scipy.linalg import sqrtm


def get_channel(args,txSim,rxSim):
    RTx = np.zeros((args['M1'] * args['M2'], args['M1'] * args['M2']))
    RRx = np.zeros((args['N1'] * args['N2'], args['N1'] * args['N2']))

    for m1 in range(args['M1'] * args['M2']):
        for m2 in range(args['M1'] * args['M2']):
            rowl1, coll1 = np.unravel_index(m1, (args['M1'], args['M2']))
            rowl2, coll2 = np.unravel_index(m2, (args['M1'], args['M2']))
            r_val = np.sqrt((txSim['XTxSim'][rowl1] - txSim['XTxSim'][rowl2]) ** 2 + (txSim['YTxSim'][coll1] - txSim['YTxSim'][coll2]) ** 2)
            RTx[m1, m2] = np.sinc(2 * r_val / args['Lam'])

    for n1 in range(args['N1'] * args['N2']):
        for n2 in range(args['N1'] * args['N2']):
            rowl1, coll1 = np.unravel_index(n1, (args['N1'], args['N2']))
            rowl2, coll2 = np.unravel_index(n2, (args['N1'], args['N2']))
            t_val = np.sqrt((rxSim['XRxSim'][rowl1] - rxSim['XRxSim'][rowl2]) ** 2 + (rxSim['YRxSim'][coll1] - rxSim['YRxSim'][coll2]) ** 2)
            RRx[n1, n2] = np.sinc(2 * t_val / args['Lam'])

    G=[]
    for j in range(args['J']):
        PL = 20 * math.log10(4 * math.pi / args['Lam'] )+10 * args['Ple'] * math.log10(args['D_tx_rx'][j])+args['Plsfv_dB']
        rho_sq = 1 / 10 ** (PL / 10)
        random_matrix = np.random.randn(args['N1'] * args['N2'], args['M1'] * args['M2']) + 1j * np.random.randn(args['N1'] * args['N2'], args['M1'] * args['M2'])
        GTilde = np.sqrt(rho_sq / 2) * random_matrix
        G.append(sqrtm(RRx) @ GTilde @ sqrtm(RTx))
    return G


def Channel(x_transmit, G, J):
    x_transmit = torch.transpose(x_transmit.float(), 1, 0)
    Y_receive = []
    for j in range(J):
        hr = G[j].real
        hi = G[j].imag
        h1 = np.concatenate([hr, -hi], 1)
        h2 = np.concatenate([hi, hr], 1)
        G1 = torch.from_numpy(np.concatenate([h1, h2], 0)).float()
        G1 = G1.cuda()
        Y_receive.append(torch.transpose(torch.matmul(G1, x_transmit), 1, 0))
    return Y_receive
