import numpy as np

def get_txSim(args):
    XTxSim = np.linspace(-args['Lam'] / 2 * (args['M1'] - 1) / 2, args['Lam'] / 2 * (args['M1'] - 1) / 2, args['M1'])
    YTxSim = np.linspace(-args['Lam'] / 2 * (args['M2'] - 1) / 2, args['Lam'] / 2 * (args['M2'] - 1) / 2, args['M2'])
    ZTxSim = np.linspace(0, args['D_tx'], args['L'])

    DTxSim = np.zeros((args['M'], args['M'], args['L'] - 1))
    ThetaTxSim = np.zeros((args['M'], args['M'], args['L'] - 1))
    WTxSim = np.zeros((args['M'], args['M'], args['L'] - 1),dtype=np.complex128)
    for l in range(args['L']-1):
        for i in range(args['M']):
            for j in range(args['M']):
                rowl1, coll1 = np.unravel_index(i, (args['M1'], args['M2']), order='F')
                rowl2, coll2 = np.unravel_index(j, (args['M1'], args['M2']), order='F')
                DTxSim[j, i, l] = np.sqrt((XTxSim[rowl1] - XTxSim[rowl2]) ** 2 + (YTxSim[coll1] - YTxSim[coll2]) ** 2 + args['dzTx'] ** 2)
                ThetaTxSim[j, i, l] = np.arccos( args['dzTx'] / np.sqrt((XTxSim[rowl1] - XTxSim[rowl2]) ** 2 + (YTxSim[coll1] - YTxSim[coll2]) ** 2 + args['dzTx'] ** 2))
                WTxSim[j, i, l] = args['dxTx'] * args['dyTx'] * np.cos(ThetaTxSim[j, i, l]) / DTxSim[j, i, l] * (1 / (2 * np.pi * DTxSim[j, i, l]) - 1j /args['Lam']) * np.exp(1j * 2 * np.pi * DTxSim[j, i, l] / args['Lam'] )


    XTxA = np.linspace(-args['Lam'] / 2 * (args['A_tx1'] - 1) / 2, args['Lam'] / 2 * (args['A_tx1'] - 1) / 2, args['A_tx1'])
    YTxA = np.linspace(-args['Lam'] / 2 * (args['A_tx2'] - 1) / 2, args['Lam'] / 2 * (args['A_tx2'] - 1) / 2, args['A_tx2'])
    DTxA = np.zeros((args['M'], args['A_tx']))
    ThetaTxA = np.zeros((args['M'], args['A_tx']))
    WTxA = np.zeros((args['M'], args['A_tx']),dtype=np.complex128)

    for i in range(args['A_tx']):
        for j in range(args['M']):
            rowl1, coll1 = np.unravel_index(i, (args['A_tx1'], args['A_tx2']), order='F')
            rowl2, coll2 = np.unravel_index(j, (args['M1'], args['M2']), order='F')
            DTxA[j, i] = np.sqrt((XTxSim[rowl2] - XTxA[rowl1]) ** 2 + (YTxSim[coll2] - YTxA[coll1]) ** 2 + args['dzTx'] ** 2)
            ThetaTxA[j, i] = np.arccos(args['dzTx'] / np.sqrt((XTxA[rowl1] - XTxSim[rowl2]) ** 2 + (YTxA[coll1] - YTxSim[coll2]) ** 2 + args['dzTx'] ** 2))
            WTxA[j, i] = args['dxTx'] * args['dyTx'] * np.cos(ThetaTxA[j, i]) / DTxA[j, i] * (1 / (2 * np.pi * DTxA[j, i]) - 1j / args['Lam']) * np.exp(1j * 2 * np.pi * DTxA[j, i] / args['Lam'])

    txSim = {
        'XTxSim': XTxSim,
        'YTxSim': YTxSim,
        'ZTxSim': ZTxSim,
        'WTxSim': WTxSim,
        'WTxA': WTxA
    }
    return txSim

def get_rxSim(args):
    XRxSim = np.linspace(-args['Lam'] / 2 * (args['N1'] - 1) / 2, args['Lam'] / 2 * (args['N1'] - 1) / 2, args['N1'])
    YRxSim = np.linspace(-args['Lam'] / 2 * (args['N2'] - 1) / 2, args['Lam'] / 2 * (args['N2'] - 1) / 2, args['N2'])
    ZRxSim = np.linspace(0, args['D_rx'], args['K'])

    DRxSim = np.zeros((args['N'], args['N'], args['K'] - 1))
    ThetaRxSim = np.zeros((args['N'], args['N'], args['K'] - 1))
    WRxSim = np.zeros((args['N'], args['N'], args['K'] - 1),dtype=np.complex128)
    for k in range(args['K']-1):
        for i in range(args['N']):
            for j in range(args['N']):
                rowl1, coll1 = np.unravel_index(i, (args['N1'], args['N2']), order='F')
                rowl2, coll2 = np.unravel_index(j, (args['N1'], args['N2']), order='F')
                DRxSim[j, i, k] = np.sqrt((XRxSim[rowl1] - XRxSim[rowl2]) ** 2 + (YRxSim[coll1] - YRxSim[coll2]) ** 2 + args['dzRx'] ** 2)
                ThetaRxSim[j, i, k] = np.arccos( args['dzRx'] / np.sqrt((XRxSim[rowl1] - XRxSim[rowl2]) ** 2 + (YRxSim[coll1] - YRxSim[coll2]) ** 2 + args['dzRx'] ** 2))
                WRxSim[j, i, k] = args['dxRx'] * args['dyRx'] * np.cos(ThetaRxSim[j, i, k]) / DRxSim[j, i, k] * (1 / (2 * np.pi * DRxSim[j, i, k]) - 1j /args['Lam']) * np.exp(1j * 2 * np.pi * DRxSim[j, i, k] / args['Lam'] )


    XRxA = np.linspace(-args['Lam'] / 2 * (args['A_rx1'] - 1) / 2, args['Lam'] / 2 * (args['A_rx1'] - 1) / 2, args['A_rx1'])
    YRxA = np.linspace(-args['Lam'] / 2 * (args['A_rx2'] - 1) / 2, args['Lam'] / 2 * (args['A_rx2'] - 1) / 2, args['A_rx2'])
    DRxA = np.zeros((args['A_rx'],args['N']))
    ThetaRxA = np.zeros((args['A_rx'],args['N']))
    WRxA = np.zeros((args['A_rx'],args['N']),dtype=np.complex128)

    for i in range(args['N']):
        for j in range(args['A_rx']):
            rowl1, coll1 = np.unravel_index(i, (args['N1'], args['N2']), order='F')
            rowl2, coll2 = np.unravel_index(j, (args['A_rx1'], args['A_rx2']), order='F')
            DRxA[j, i] = np.sqrt((XRxSim[rowl1] - XRxA[rowl2]) ** 2 + (YRxSim[rowl1] - YRxA[coll2]) ** 2 + args['dzRx'] ** 2)
            ThetaRxA[j, i] = np.arccos(args['dzRx'] / np.sqrt((XRxA[rowl2] - XRxSim[rowl1] ) ** 2 + (YRxA[coll2] - YRxSim[rowl1]) ** 2 + args['dzRx'] ** 2))
            WRxA[j, i] = args['dxRx'] * args['dyRx'] * np.cos(ThetaRxA[j, i]) / DRxA[j, i] * (1 / (2 * np.pi * DRxA[j, i]) - 1j / args['Lam']) * np.exp(1j * 2 * np.pi * DRxA[j, i] / args['Lam'])

    rxSim = {
        'XRxSim': XRxSim,
        'YRxSim': YRxSim,
        'ZRxSim': ZRxSim,
        'WRxSim': WRxSim,
        'WRxA': WRxA
    }
    
    return rxSim
