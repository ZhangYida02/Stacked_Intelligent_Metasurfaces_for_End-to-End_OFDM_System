import numpy as np

def get_txDpSim(args):
    XTxDpSim = np.linspace(-args['Lam_c'] / 2 * (args['M1_dp'] - 1) / 2, args['Lam_c'] / 2 * (args['M1_dp'] - 1) / 2, args['M1_dp'])
    YTxDpSim = np.linspace(-args['Lam_c'] / 2 * (args['M2_dp'] - 1) / 2, args['Lam_c'] / 2 * (args['M2_dp'] - 1) / 2, args['M2_dp'])
    ZTxDpSim = np.linspace(args['dzTx'], args['dzTx'] * args['L'], args['L'])

    DTxDpSim = np.zeros((args['M_dp'], args['M_dp'], args['L'] - 1))
    ThetaTxDpSim = np.zeros((args['M_dp'], args['M_dp'], args['L'] - 1))
    WTxSim = np.zeros((args['M_dp'], args['M_dp'], args['L'] - 1, args['N_c']),dtype=np.complex64)
    for l in range(args['L']-1):
        for i in range(args['M_dp']):
            for j in range(args['M_dp']):
                rowl1, coll1 = np.unravel_index(i, (args['M1_dp'], args['M2_dp']), order='F')
                rowl2, coll2 = np.unravel_index(j, (args['M1_dp'], args['M2_dp']), order='F')
                DTxDpSim[j, i, l] = np.sqrt((XTxDpSim[rowl1] - XTxDpSim[rowl2]) ** 2 + (YTxDpSim[coll1] - YTxDpSim[coll2]) ** 2 + args['dzTx'] ** 2)
                ThetaTxDpSim[j, i, l] = np.arccos( args['dzTx'] / np.sqrt((XTxDpSim[rowl1] - XTxDpSim[rowl2]) ** 2 + (YTxDpSim[coll1] - YTxDpSim[coll2]) ** 2 + args['dzTx'] ** 2))
                for n_c in range(args['N_c']):
                    WTxSim[j, i, l, n_c] = args['dxTx'] * args['dyTx'] * np.cos(ThetaTxDpSim[j, i, l]) / DTxDpSim[j, i, l] * (1 / (2 * np.pi * DTxDpSim[j, i, l]) - 1j /args['Lam'][n_c]) * np.exp(1j * 2 * np.pi * DTxDpSim[j, i, l] / args['Lam'][n_c] )


    XTxA = np.linspace(-args['Lam_c'] / 2 * (args['A_tx1_dp'] - 1) / 2, args['Lam_c'] / 2 * (args['A_tx1_dp'] - 1) / 2, args['A_tx1_dp'])
    YTxA = np.linspace(-args['Lam_c'] / 2 * (args['A_tx2_dp'] - 1) / 2, args['Lam_c'] / 2 * (args['A_tx2_dp'] - 1) / 2, args['A_tx2_dp'])
    DTxA = np.zeros((args['M_dp'], args['A_tx_dp']))
    ThetaTxA = np.zeros((args['M_dp'], args['A_tx_dp']))
    WTxA = np.zeros((args['M_dp'], args['A_tx_dp'], args['N_c']),dtype=np.complex64)

    for i in range(args['A_tx_dp']):
        for j in range(args['M_dp']):
            rowl1, coll1 = np.unravel_index(i, (args['A_tx1_dp'], args['A_tx2_dp']), order='F')
            rowl2, coll2 = np.unravel_index(j, (args['M1_dp'], args['M2_dp']), order='F')
            DTxA[j, i] = np.sqrt((XTxDpSim[rowl2] - XTxA[rowl1]) ** 2 + (YTxDpSim[coll2] - YTxA[coll1]) ** 2 + args['dzTx'] ** 2)
            ThetaTxA[j, i] = np.arccos(args['dzTx'] / np.sqrt((XTxA[rowl1] - XTxDpSim[rowl2]) ** 2 + (YTxA[coll1] - YTxDpSim[coll2]) ** 2 + args['dzTx'] ** 2))
            for n_c in range(args['N_c']):
                WTxA[j, i, n_c] = args['dxTx'] * args['dyTx'] * np.cos(ThetaTxA[j, i]) / DTxA[j, i] * (1 / (2 * np.pi * DTxA[j, i]) - 1j / args['Lam'][n_c]) * np.exp(1j * 2 * np.pi * DTxA[j, i] / args['Lam'][n_c])


    WTxDpA = np.zeros((2 * WTxA.shape[0], 2 * WTxA.shape[1], WTxA.shape[2]), dtype=WTxA.dtype)
    WTxDpA[:WTxA.shape[0], :WTxA.shape[1], :] = WTxA
    WTxDpA[WTxA.shape[0]:, WTxA.shape[1]:, :] = WTxA

    WTxDpSim = np.zeros((2 * WTxSim.shape[0], 2 * WTxSim.shape[1], WTxSim.shape[2], WTxSim.shape[3]), dtype=WTxSim.dtype)
    WTxDpSim[:WTxSim.shape[0], :WTxSim.shape[1], :, :] = WTxSim
    WTxDpSim[WTxSim.shape[0]:, WTxSim.shape[1]:, :, :] = WTxSim

    txDpSim = {
        'XTxDpSim': XTxDpSim,
        'YTxDpSim': YTxDpSim,
        'ZTxDpSim': ZTxDpSim,
        'WTxDpSim': WTxDpSim,
        'WTxDpA': WTxDpA
    }
    return txDpSim

def get_rxDpSim(args):
    XRxDpSim = np.linspace(-args['Lam_c'] / 2 * (args['N1_dp'] - 1) / 2, args['Lam_c'] / 2 * (args['N1_dp'] - 1) / 2, args['N1_dp'])
    YRxDpSim = np.linspace(-args['Lam_c'] / 2 * (args['N2_dp'] - 1) / 2, args['Lam_c'] / 2 * (args['N2_dp'] - 1) / 2, args['N2_dp'])
    ZRxDpSim = np.linspace(args['dzRx'], args['dzRx'] * args['K'], args['K'])

    DRxDpSim = np.zeros((args['N_dp'], args['N_dp'], args['K'] - 1))
    ThetaRxDpSim = np.zeros((args['N_dp'], args['N_dp'], args['K'] - 1))
    WRxSim = np.zeros((args['N_dp'], args['N_dp'], args['K'] - 1, args['N_c']),dtype=np.complex64)
    for k in range(args['K']-1):
        for i in range(args['N_dp']):
            for j in range(args['N_dp']):
                rowl1, coll1 = np.unravel_index(i, (args['N1_dp'], args['N2_dp']), order='F')
                rowl2, coll2 = np.unravel_index(j, (args['N1_dp'], args['N2_dp']), order='F')
                DRxDpSim[j, i, k] = np.sqrt((XRxDpSim[rowl1] - XRxDpSim[rowl2]) ** 2 + (YRxDpSim[coll1] - YRxDpSim[coll2]) ** 2 + args['dzRx'] ** 2)
                ThetaRxDpSim[j, i, k] = np.arccos( args['dzRx'] / np.sqrt((XRxDpSim[rowl1] - XRxDpSim[rowl2]) ** 2 + (YRxDpSim[coll1] - YRxDpSim[coll2]) ** 2 + args['dzRx'] ** 2))
                for n_c in range(args['N_c']):
                    WRxSim[j, i, k, n_c] = args['dxRx'] * args['dyRx'] * np.cos(ThetaRxDpSim[j, i, k]) / DRxDpSim[j, i, k] * (1 / (2 * np.pi * DRxDpSim[j, i, k]) - 1j /args['Lam'][n_c]) * np.exp(1j * 2 * np.pi * DRxDpSim[j, i, k] / args['Lam'][n_c] )


    XRxA = np.linspace(-args['Lam_c'] / 2 * (args['A_rx1_dp'] - 1) / 2, args['Lam_c'] / 2 * (args['A_rx1_dp'] - 1) / 2, args['A_rx1_dp'])
    YRxA = np.linspace(-args['Lam_c'] / 2 * (args['A_rx2_dp'] - 1) / 2, args['Lam_c'] / 2 * (args['A_rx2_dp'] - 1) / 2, args['A_rx2_dp'])
    DRxA = np.zeros((args['A_rx_dp'], args['N_dp']))
    ThetaRxA = np.zeros((args['A_rx_dp'], args['N_dp']))
    WRxA = np.zeros((args['A_rx_dp'], args['N_dp'], args['N_c']), dtype=np.complex64)

    for i in range(args['N_dp']):
        for j in range(args['A_rx_dp']):
            rowl1, coll1 = np.unravel_index(i, (args['N1_dp'], args['N2_dp']), order='F')
            rowl2, coll2 = np.unravel_index(j, (args['A_rx1_dp'], args['A_rx2_dp']), order='F')
            DRxA[j, i] = np.sqrt((XRxDpSim[rowl1] - XRxA[rowl2]) ** 2 + (YRxDpSim[rowl1] - YRxA[coll2]) ** 2 + args['dzRx'] ** 2)
            ThetaRxA[j, i] = np.arccos(args['dzRx'] / np.sqrt((XRxA[rowl2] - XRxDpSim[rowl1] ) ** 2 + (YRxA[coll2] - YRxDpSim[rowl1]) ** 2 + args['dzRx'] ** 2))
            for n_c in range(args['N_c']):
                WRxA[j, i, n_c] = args['dxRx'] * args['dyRx'] * np.cos(ThetaRxA[j, i]) / DRxA[j, i] * (1 / (2 * np.pi * DRxA[j, i]) - 1j / args['Lam'][n_c]) * np.exp(1j * 2 * np.pi * DRxA[j, i] / args['Lam'][n_c])

    WRxDpA = np.zeros((2 * WRxA.shape[0], 2 * WRxA.shape[1], WRxA.shape[2]), dtype=WRxA.dtype)
    WRxDpA[:WRxA.shape[0], :WRxA.shape[1], :] = WRxA
    WRxDpA[WRxA.shape[0]:, WRxA.shape[1]:, :] = WRxA

    WRxDpSim = np.zeros((2 * WRxSim.shape[0], 2 * WRxSim.shape[1], WRxSim.shape[2], WRxSim.shape[3]), dtype=WRxSim.dtype)
    WRxDpSim[:WRxSim.shape[0], :WRxSim.shape[1], :, :] = WRxSim
    WRxDpSim[WRxSim.shape[0]:, WRxSim.shape[1]:, :, :] = WRxSim

    rxDpSim = {
        'XRxDpSim': XRxDpSim,
        'YRxDpSim': YRxDpSim,
        'ZRxDpSim': ZRxDpSim,
        'WRxDpSim': WRxDpSim,
        'WRxDpA': WRxDpA
    }

    return rxDpSim
