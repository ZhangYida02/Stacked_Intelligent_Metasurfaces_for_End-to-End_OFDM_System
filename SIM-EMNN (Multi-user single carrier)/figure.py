import matplotlib.pyplot as plt

def plot_ber (Noise_test_dBm,ber):
    plt.figure(2)
    plt.plot(Noise_test_dBm, ber)
    plt.yscale('log')
    plt.xlabel('Received noise (dBm)')
    plt.ylabel('BER')
    plt.ylim(10**-6, 1)
    plt.grid(True, which="both", linestyle='--')

