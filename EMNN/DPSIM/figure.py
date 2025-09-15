import matplotlib.pyplot as plt

def plot_ber (Noise_test_dBm,ber):
    plt.figure(2)
    plt.plot(Noise_test_dBm, ber)
    plt.yscale('log')
    plt.xlabel('P_tx_test_dBm (dBm)')
    plt.ylabel('BER')
    plt.ylim(10**-6, 1)
    plt.xlim(-20, 50)
    plt.grid(True, which="both", linestyle='--')

def plot_loss(R_error):
    plt.figure(1)
    plt.plot(R_error)
    plt.ylim(bottom=0)

