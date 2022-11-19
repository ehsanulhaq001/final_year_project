import matplotlib.pyplot as plt
import numpy as np
from scipy import special


def q_function(x):
    return 0.5-0.5*special.erf(x/np.sqrt(2))


def generate_noise(signal, Eb_N0):
    """Generates Noise for given size and Power"""

    ns = len(signal)
    noise = (1 / np.sqrt(2 * Eb_N0)) * (
        np.random.normal(size=ns) + np.random.normal(size=ns) * 1j
    )
    return noise


def generate_QPSK_BER_vs_EbN0_Pr(Eb_N0_dB_range):
    """Generates BER vs Eb/N0 for QPSK"""

    msg = np.random.randint(low=0, high=2, size=int(1e5))

    symbols = np.array([msg[0::2], msg[1::2]])
    signal = []

    for k in range(np.size(symbols, axis=1)):
        b_0 = symbols[0, k]
        b_1 = symbols[1, k]
        if b_0 == 0 and b_1 == 0:
            theta = 5.0 * np.math.pi / 4.0
            signal.append(np.array(np.cos(theta) + 1j * np.sin(theta)))
        elif b_0 == 0 and b_1 == 1:
            theta = 3.0 * np.math.pi / 4.0
            signal.append(np.array(np.cos(theta) + 1j * np.sin(theta)))
        elif b_0 == 1 and b_1 == 1:
            theta = 1.0 * np.math.pi / 4.0
            signal.append(np.array(np.cos(theta) + 1j * np.sin(theta)))
        elif b_0 == 1 and b_1 == 0:
            theta = 7.0 * np.math.pi / 4.0
            signal.append(np.array(np.cos(theta) + 1j * np.sin(theta)))

    BER_wireless = []
    BER_AWGN = []
    for Eb_N0_dB in Eb_N0_dB_range:
        Eb_N0 = 10 ** (Eb_N0_dB / 10.0)
        noise = generate_noise(signal, Eb_N0)

        h = (1 / np.sqrt(2)) * (
            np.random.normal(size=len(signal)) + 1j *
            np.random.normal(size=len(signal))
        )

        y = h * signal + noise
        y /= h

        y_2 = signal + noise
        wirelessly_received_msg = []
        received_msg = []

        for i in y:
            wirelessly_received_msg.append(np.real(i) > 0)
            wirelessly_received_msg.append(np.imag(i) > 0)

        for i in y_2:
            received_msg.append(np.real(i) > 0)
            received_msg.append(np.imag(i) > 0)

        Pb_pr_wireless = np.count_nonzero(
            msg != wirelessly_received_msg) / len(msg)
        Pb_pr = np.count_nonzero(msg != received_msg) / len(msg)
        BER_AWGN.append(Pb_pr)
        BER_wireless.append(Pb_pr_wireless)

    return BER_AWGN, BER_wireless


EbN0dB_range = range(0, 21)
BER_AWGN, BER_wireless = generate_QPSK_BER_vs_EbN0_Pr(EbN0dB_range)

snr = 10 ** (np.array(EbN0dB_range) / 10.0)

BER_AWGN_ideal = 1/2 * (1 - np.sqrt(snr/(snr+2)))

BER_wireless_ideal = q_function(np.sqrt(snr))

plt.plot(EbN0dB_range, BER_AWGN, "s-",
         label="QPSK over AWGN channel - Simulations")
plt.plot(
    EbN0dB_range,
    BER_wireless,
    "d-",
    label="QPSK over Rayleigh channel - Simulations",
)
plt.plot(EbN0dB_range, BER_AWGN_ideal, "--",
         label="QPSK over Rayleigh channel - Theoretical")

plt.plot(EbN0dB_range, BER_wireless_ideal, "--",
         label="QPSK over AWGN channel - Theoretical")
plt.ylim(bottom=10**(-6))
plt.xlabel("SNR (dB)")
plt.xscale("linear")
plt.ylabel("BER")
plt.yscale("log")
plt.legend()
plt.grid()
plt.show()
