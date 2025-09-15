import numpy as np





def onehot2bit(X, K_bit):
    bit = np.zeros((X.shape[0], K_bit))
    for i in range(X.shape[0]):
        str = format(X[i,0], f'0{K_bit}b')
        for k in range(K_bit):
            bit[i, k] = int(str[k])
    return bit

def BER(Y_vali, y_pred, args):
    num = y_pred.shape[0]
    y_pred[y_pred<0.5] = 0
    y_pred[y_pred>=0.5] = 1
    X_bit = onehot2bit(Y_vali, args['K_bit'])
    err = np.sum(np.abs(y_pred - X_bit))
    ber = err / (num * args['K_bit'])
    return ber

def generate_rate_data(M, J):
    X_symbol_index = np.tile(np.arange(M), (J, 1)).T
    Y_symbol_index = X_symbol_index
    return X_symbol_index, Y_symbol_index

def generate_transmit_data(M, num):
    X_symbol_index = np.random.randint(M,size=(num, 1), dtype=np.int64)
    Y_symbol_index = X_symbol_index
    return X_symbol_index, Y_symbol_index

def calcul_rate(y, Noise_test_dbm, Num_Antenna):
    Noise_test = 10 ** (Noise_test_dbm / 10) * 10 ** - 3
    z = y[:,0:Num_Antenna] + 1j*y[:,Num_Antenna:2*Num_Antenna]
    zt = y[:,0:Num_Antenna] - 1j*y[:,Num_Antenna:2*Num_Antenna]
    rate = np.log2(1 + np.matmul(z, zt.T)/Noise_test)
    mean_rate = np.mean(np.diag(rate))
    max_rate = np.max(np.diag(rate))
    return abs(mean_rate), abs(max_rate)