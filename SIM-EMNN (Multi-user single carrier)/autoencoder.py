import numpy as np





def onehot2bit(X, J, K_bit):
    bit = np.zeros((X.shape[0], J*K_bit))
    for i in range(X.shape[0]):
        for j in range(J):
            str = format(X[i, j], f'0{K_bit}b')
            for k in range(K_bit):
                bit[i, j*K_bit+k] = int(str[k])
    """
    BIT_16 = np.array([[0,0,0,0],[0,0,0,1],[0,0,1,0],[0,0,1,1],
                       [0,1,0,0],[0,1,0,1],[0,1,1,0],[0,1,1,1],
                       [1,0,0,0],[1,0,0,1],[1,0,1,0],[1,0,1,1],
                       [1,1,0,0],[1,1,0,1],[1,1,1,0],[1,1,1,1]])
    bit = np.matmul(X, BIT_16)
    """
    return bit
    
def BER(X, y_pred, args):
    err = 0
    num = y_pred.shape[0]
    y_pred[y_pred<0.5] = 0
    y_pred[y_pred>=0.5] = 1
    y_pred = np.reshape(y_pred,[num*args['J'],args['K_bit']])
    X_bit = onehot2bit(X, args['J'], args['K_bit'])
    X_bit = np.reshape(X_bit,[num*args['J'],args['K_bit']])
    err = np.sum(np.abs(y_pred - X_bit))
    ber = err / (num * args['J'] * args['K_bit'])
    return ber

def generate_rate_data(M, J):
    X_symbol_index = np.tile(np.arange(M), (J, 1)).T
    Y_symbol_index = X_symbol_index
    return X_symbol_index, Y_symbol_index

def generate_transmit_data(M, J, num, seed=0):
    np.random.seed(seed)
    X_symbol_index = np.random.randint(M,size=(num,J))
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