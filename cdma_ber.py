import numpy as np
import matplotlib.pyplot as plt
from commpy.sequences import pnsequence

def generate_noise(ns, snr):
    noise = (1 / np.sqrt(snr)) * np.random.normal(size=ns)
    return noise

# generate binary string of alternate 1s and 0s of size l


def generate_alternate_binary_string(l):
    seq = np.zeros(l)
    seq[0::2] = 1
    return seq


def generate_pn_sequence(pn_order):
    pn_seq = pnsequence(pn_order, generate_alternate_binary_string(
        pn_order), generate_alternate_binary_string(pn_order), 2**pn_order-1)
    pn_seq[pn_seq == 0] = -1
    return pn_seq

# generate a PN sequence
# pn_order = 4
# pn_seq = pnsequence(pn_order, generate_alternate_binary_string(
#     pn_order), generate_alternate_binary_string(pn_order), 2**pn_order-1)
# pn_seq[pn_seq == 0] = -1

# pn_seq = generate_pn_sequence(8)

# if (1):
#     # check Balancing Property
#     print((1-np.sum(pn_seq)/len(pn_seq))*100, "% balanced")

#     # check Autocorrelation Property
#     autocorrelation = []
#     for shift in range(0, len(pn_seq)-1):

#         correlation = np.corrcoef(pn_seq, np.roll(pn_seq, shift))[0, 1]

#         autocorrelation.append(correlation)

#     plt.plot(autocorrelation)
#     plt.show()


def generate_BER_CDMA(MESSAGE_SIZE, SNR_DB_RANGE, pn_order, fading=False):
    BER_CDMA = []
    # generate Message signal
    message_1 = np.random.randint(
        0, 2, MESSAGE_SIZE).reshape((MESSAGE_SIZE, 1))
    message_1[message_1 == 0] = -1
    message_2 = np.random.randint(
        0, 2, MESSAGE_SIZE).reshape((MESSAGE_SIZE, 1))
    message_2[message_2 == 0] = -1

    # generate PN sequence
    pn_seq_1 = generate_pn_sequence(pn_order).reshape((2**pn_order-1, 1))
    pn_seq_2 = np.roll(pn_seq_1, 34).reshape((2**pn_order-1, 1))

    for snr_db in SNR_DB_RANGE:
        # print(snr_db, end="\r")
        snr = 10**(snr_db/10)
        print("snr ", round(snr, 3), end="\t")

        # generate Transmitted Signal
        transmitted_signal_1 = np.dot(message_1, pn_seq_1.T)
        transmitted_signal_2 = np.dot(message_2, pn_seq_2.T)

        # flatter 2d array
        transmitted_signal_1 = transmitted_signal_1.flatten()
        transmitted_signal_2 = transmitted_signal_2.flatten()
        transmitted_signal = transmitted_signal_1 + transmitted_signal_2

        # generate Noise
        noise = generate_noise(len(transmitted_signal), snr)

        print("noise: N(", round(np.mean(noise), 3),
              ",", round(np.var(noise), 3), ")", end="\t")

        h = np.ones(len(transmitted_signal))
        if (fading):
            h = 1 - np.random.normal(size=len(transmitted_signal))/2

        print("fading: N(", round(np.mean(h), 3),
              ",", round(np.var(h), 3), ")")

        # generate Received Signal
        received_signal = h * transmitted_signal + noise
        received_signal = received_signal.reshape(
            (MESSAGE_SIZE, 2**pn_order-1))

        # generate Received Message

        received_message_1 = np.dot(received_signal, pn_seq_1)/16
        received_message_2 = np.dot(received_signal, pn_seq_2)/16

        received_message_1[received_message_1 > 0] = 1
        received_message_1[received_message_1 < 0] = -1
        received_message_2[received_message_2 > 0] = 1
        received_message_2[received_message_2 < 0] = -1

        ber_1 = np.sum(received_message_1 != message_1)/MESSAGE_SIZE
        ber_2 = np.sum(received_message_2 != message_2)/MESSAGE_SIZE

        ber_avg = max(ber_1, ber_2)
        BER_CDMA.append(ber_avg)

    return BER_CDMA


MESSAGE_SIZE = 10**5
SNR_DB_RANGE = range(-20, 5)

BER_CDMA_1 = generate_BER_CDMA(MESSAGE_SIZE, SNR_DB_RANGE, 4)
BER_CDMA_2 = generate_BER_CDMA(MESSAGE_SIZE, SNR_DB_RANGE, 4, True)
BER_CDMA_3 = generate_BER_CDMA(MESSAGE_SIZE, SNR_DB_RANGE, 3)
BER_CDMA_4 = generate_BER_CDMA(MESSAGE_SIZE, SNR_DB_RANGE, 3, True)


plt.plot(SNR_DB_RANGE, BER_CDMA_1, "d-", label="CDMA 4 no fading")
plt.plot(SNR_DB_RANGE, BER_CDMA_2, "*-", label="CDMA 4 fading")
plt.plot(SNR_DB_RANGE, BER_CDMA_3, "d-", label="CDMA 3 no fading")
plt.plot(SNR_DB_RANGE, BER_CDMA_4, "*-", label="CDMA 3 fading")

plt.xlabel("SNR (dB)")
plt.xscale("linear")
plt.ylabel("BER")
plt.yscale("log")
plt.legend()
plt.grid()
plt.show()