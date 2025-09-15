import math
import numpy as np
import torch
import os
import scipy.io as sio
from pathlib import Path
from scipy.linalg import sqrtm


def get_channel(args):
    print(f"信道已从 {args['channel_path']} 加载")
    data = sio.loadmat(args['channel_path'])
    G= data['G']
    return G




def Channel(x_transmit, G, J, args):
    m,n=x_transmit.shape
    x_transmit = torch.transpose(x_transmit.cfloat(), 1, 0)
    Y_receive = []
    for j in range(J):
        y_receive = torch.from_numpy(np.zeros((2 * args['N_dp'] * args['N_c'], m), dtype=np.complex64)).cuda()
        for n_c in range(args['N_c']):
            G1=torch.from_numpy(G[:,:,:,j]).cuda().cfloat()
            y_receive_single=G1[:,:,n_c]@ x_transmit[n_c * 2 * args['M_dp']: (n_c + 1) * 2 * args['M_dp'], :]
            y_receive[n_c * 2 * args['N_dp']: (n_c + 1) * 2 * args['N_dp'], :]=y_receive_single
        Y_receive.append(torch.transpose(y_receive, 1, 0))
    return Y_receive
