import scipy.signal as sig
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import numpy as np
import time
import math
import pandas as pd
import tkinter as tk
from tkinter import ttk
from skimage.metrics import peak_signal_noise_ratio, mean_squared_error
from PyEMD import EMD

# wziete ze starej dokumentacji scipy
def signaltonoise(a, axis=0, ddof=0):
    """
    The signal-to-noise ratio of the input data.

    Returns the signal-to-noise ratio of `a`, here defined as the mean
    divided by the standard deviation.

    Parameters
    ----------
    a : array_like
        An array_like object containing the sample data.
    axis : int or None, optional
        If axis is equal to None, the array is first ravel'd. If axis is an
        integer, this is the axis over which to operate. Default is 0.
    ddof : int, optional
        Degrees of freedom correction for standard deviation. Default is 0.

    Returns
    -------
    s2n : ndarray
        The mean to standard deviation ratio(s) along `axis`, or 0 where the
        standard deviation is 0.

    """
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m/sd)

def zad1(amplituda, frekwencja, czest_prob, t_trwania, n):
    t = np.linspace(0, t_trwania, int(czest_prob * t_trwania), endpoint=False)
    signal = amplituda * sig.chirp(2 * np.pi * frekwencja * t, 6, 10, 1)
    szum = np.random.normal(0, n, len(signal))


    signal_power = np.sum(np.square(signal))
    noise_power = np.sum(np.square(szum))

    # SNR
    snr = 20 * np.log10(signal_power / noise_power)
    print("SNR:", snr)

    # MSE
    mse = np.mean((signal - szum) ** 2)
    print("MSE:", mse)

    # PSNR
    psnr = 20 * np.log10(np.abs(signal).max() / np.sqrt(mse))
    print("PSNR:", psnr)

    print("\n\n")
def z1():
    root = tk.Tk()
    root.title("Manipulacja parametrami")

    label_amplituda = tk.Label(root, text='Amplituda:')
    label_amplituda.grid(row=0, column=0)
    slider_amplituda = tk.Scale(root, from_=0.1, to=2.0, orient=tk.HORIZONTAL, resolution=0.1)
    slider_amplituda.grid(row=0, column=1)

    label_frekwencja = tk.Label(root, text='Częstotliwość:')
    label_frekwencja.grid(row=1, column=0)
    slider_frekwencja = tk.Scale(root, from_=1.0, to=10.0, orient=tk.HORIZONTAL, resolution=0.5)
    slider_frekwencja.grid(row=1, column=1)

    label_czest_prob = tk.Label(root, text='Częstotliwość próbkowania:')
    label_czest_prob.grid(row=2, column=0)
    slider_czest_prob = tk.Scale(root, from_=50, to=200, orient=tk.HORIZONTAL, resolution=10)
    slider_czest_prob.grid(row=2, column=1)

    label_t_trwania = tk.Label(root, text='Czas trwania:')
    label_t_trwania.grid(row=3, column=0)
    slider_t_trwania = tk.Scale(root, from_=0.5, to=2.0, orient=tk.HORIZONTAL, resolution=0.1)
    slider_t_trwania.grid(row=3, column=1)

    label_szum = tk.Label(root, text='amplituda szumu:')
    label_szum.grid(row=4, column=0)
    slider_szum = tk.Scale(root, from_=0, to=5.0, orient=tk.HORIZONTAL, resolution=0.1)
    slider_szum.grid(row=4, column=1)

    button_generate = ttk.Button(root, text="Generuj sygnał",
                                 command=lambda: zad1(slider_amplituda.get(), slider_frekwencja.get(),
                                                      slider_czest_prob.get(), slider_t_trwania.get(),
                                                      slider_szum.get()))
    button_generate.grid(row=5, columnspan=2)

    root.mainloop()


def zad2(amplituda, frekwencja, czest_prob, t_trwania, n):
    t = np.linspace(0, t_trwania, int(czest_prob * t_trwania), endpoint=False)
    signal = amplituda * sig.chirp(2 * np.pi * frekwencja * t, 6, 10, 1)
    szum = np.random.normal(0, n, len(signal))

    signal_power = np.sum(signal ** 2)
    noise_power = np.sum(szum ** 2)

    # SNR
    snr = 20 * np.log10(signal_power / noise_power)

    # MSE
    mse = np.mean((signal - szum) ** 2)

    # PSNR
    psnr = 20 * np.log10(np.abs(signal).max() / np.sqrt(mse))

    return snr, mse, psnr

def compare_metrics(amplituda, frekwencja, czest_prob, t_trwania, n):
    snr_zad1, mse_zad1, psnr_zad1 = zad2(amplituda, frekwencja, czest_prob, t_trwania, n)

    t = np.linspace(0, t_trwania, int(czest_prob * t_trwania), endpoint=False)
    signal = amplituda * sig.chirp(2 * np.pi * frekwencja * t, 6, 10, 1)
    szum = np.random.normal(0, n, len(signal))

    snr_builtin = signaltonoise(signal + szum)
    mse_builtin = mean_squared_error(signal, signal + szum)
    psnr_builtin = peak_signal_noise_ratio(signal, signal + szum, data_range=np.abs(signal).max())

    print("SNR (zad1):", snr_zad1)
    print("SNR (biblioteka):", snr_builtin)
    print("MSE (zad1):", mse_zad1)
    print("MSE (biblioteka):", mse_builtin)
    print("PSNR (zad1):", psnr_zad1)
    print("PSNR (biblioteka):", psnr_builtin)
    print("\n\n")
def z2():
    root = tk.Tk()
    root.title("Manipulacja parametrami")

    label_amplituda = tk.Label(root, text='Amplituda:')
    label_amplituda.grid(row=0, column=0)
    slider_amplituda = tk.Scale(root, from_=0.1, to=2.0, orient=tk.HORIZONTAL, resolution=0.1)
    slider_amplituda.grid(row=0, column=1)

    label_frekwencja = tk.Label(root, text='Częstotliwość:')
    label_frekwencja.grid(row=1, column=0)
    slider_frekwencja = tk.Scale(root, from_=1.0, to=10.0, orient=tk.HORIZONTAL, resolution=0.5)
    slider_frekwencja.grid(row=1, column=1)

    label_czest_prob = tk.Label(root, text='Częstotliwość próbkowania:')
    label_czest_prob.grid(row=2, column=0)
    slider_czest_prob = tk.Scale(root, from_=50, to=200, orient=tk.HORIZONTAL, resolution=10)
    slider_czest_prob.grid(row=2, column=1)

    label_t_trwania = tk.Label(root, text='Czas trwania:')
    label_t_trwania.grid(row=3, column=0)
    slider_t_trwania = tk.Scale(root, from_=0.5, to=2.0, orient=tk.HORIZONTAL, resolution=0.1)
    slider_t_trwania.grid(row=3, column=1)

    label_szum = tk.Label(root, text='amplituda szumu:')
    label_szum.grid(row=4, column=0)
    slider_szum = tk.Scale(root, from_=0, to=5.0, orient=tk.HORIZONTAL, resolution=0.1)
    slider_szum.grid(row=4, column=1)

    button_generate = ttk.Button(root, text="Generuj sygnał",
                                 command=lambda: compare_metrics(slider_amplituda.get(), slider_frekwencja.get(),
                                                                  slider_czest_prob.get(), slider_t_trwania.get(),
                                                                  slider_szum.get()))
    button_generate.grid(row=5, columnspan=2)

    root.mainloop()


def zad3(amplitude, frequency, sampling_rate, duration, white_noise_std, brown_noise_std,
                                     snr_db):
    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)

    swiergot = amplitude * sig.chirp(2 * np.pi * frequency * t, 6, duration, 1)

    white = np.random.normal(0, white_noise_std, len(t))

    brown = np.cumsum(np.random.normal(0, brown_noise_std, len(t)))


    #standaryzacja
    brown -= np.mean(brown)
    brown *= (brown_noise_std / np.std(brown))
    signal_power = np.sum(np.square(swiergot))
    desired_noise_power = signal_power / (10 ** (snr_db / 10))
    current_noise_power = np.sum(np.square(white + brown))
    scaling_factor = np.sqrt(desired_noise_power / current_noise_power)
    white *= scaling_factor
    brown *= scaling_factor

    szum_syg = swiergot + white + brown


    plt.figure(figsize=(10, 6))
    plt.subplot(2,2,1)
    plt.plot(t, szum_syg, label='zaszumiony sygnal')
    plt.xlabel('czas')
    plt.ylabel('Amplituda')
    plt.title('swiergotliwy z szumami browna i bialym (SNR = {} dB)'.format(snr_db))
    plt.legend()
    plt.grid(True)

    plt.subplot(2,2,2)
    plt.plot(t, swiergot, label='czysty sygnal')
    plt.xlabel('czas')
    plt.ylabel('Amplituda')
    plt.title('swiergotliwy'.format(snr_db))
    plt.legend()
    plt.grid(True)

    plt.subplot(2,2,3)
    plt.plot(t, white, label='White szum', c = 'red')
    plt.xlabel('czas')
    plt.ylabel('Amplituda')
    plt.title('bialy szum'.format(snr_db))
    plt.legend()
    plt.grid(True)

    plt.subplot(2,2,4)
    plt.plot(t, brown, label='Brown szum' ,c='brown')
    plt.xlabel('czas')
    plt.ylabel('Amplituda')
    plt.title('szum browna'.format(snr_db))
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    zad4(szum_syg,t,snr_db,swiergot)
    zad5(szum_syg,t,snr_db,swiergot)
    zad6(szum_syg,t,snr_db,swiergot)
def z3():
    root = tk.Tk()
    root.title("Manipulacja parametrami")

    label_amplituda = tk.Label(root, text='Amplituda:')
    label_amplituda.grid(row=0, column=0)
    slider_amplituda = tk.Scale(root, from_=0.1, to=2.0, orient=tk.HORIZONTAL, resolution=0.1)
    slider_amplituda.grid(row=0, column=1)

    label_frekwencja = tk.Label(root, text='Częstotliwość:')
    label_frekwencja.grid(row=1, column=0)
    slider_frekwencja = tk.Scale(root, from_=1.0, to=10.0, orient=tk.HORIZONTAL, resolution=0.5)
    slider_frekwencja.grid(row=1, column=1)

    label_czest_prob = tk.Label(root, text='Częstotliwość próbkowania:')
    label_czest_prob.grid(row=2, column=0)
    slider_czest_prob = tk.Scale(root, from_=50, to=200, orient=tk.HORIZONTAL, resolution=10)
    slider_czest_prob.grid(row=2, column=1)

    label_t_trwania = tk.Label(root, text='Czas trwania:')
    label_t_trwania.grid(row=3, column=0)
    slider_t_trwania = tk.Scale(root, from_=0.5, to=2.0, orient=tk.HORIZONTAL, resolution=0.1)
    slider_t_trwania.grid(row=3, column=1)

    label_szum = tk.Label(root, text='odchylenie std szumu bialego:')
    label_szum.grid(row=4, column=0)
    slider_szum = tk.Scale(root, from_=0.1, to=5.0, orient=tk.HORIZONTAL, resolution=0.1)
    slider_szum.grid(row=4, column=1)

    label_szum2 = tk.Label(root, text='odchylenie std szumu browna:')
    label_szum2.grid(row=5, column=0)
    slider_szum2 = tk.Scale(root, from_=0.1, to=5.0, orient=tk.HORIZONTAL, resolution=0.1)
    slider_szum2.grid(row=5, column=1)

    label_snr = tk.Label(root, text='snr:')
    label_snr.grid(row=6, column=0)
    snr = tk.Scale(root, from_=0.1, to=10.0, orient=tk.HORIZONTAL, resolution=0.1)
    snr.grid(row=6, column=1)

    button_generate = ttk.Button(root, text="Generuj sygnał",
                                 command=lambda: zad3(slider_amplituda.get(), slider_frekwencja.get(),
                                                                  slider_czest_prob.get(), slider_t_trwania.get(),
                                                                  slider_szum.get(), slider_szum2.get(), snr.get()))
    button_generate.grid(row=7, columnspan=2)

    root.mainloop()


def zad4(szum_syg,t, snr_db,oryg):

    dsignal = sig.wiener(szum_syg)

    plt.figure(figsize=(10, 6))

    plt.subplot(3,1,1)
    plt.plot(t, szum_syg, label='zaszumiony sygnal')
    plt.xlabel('czas')
    plt.ylabel('Amplituda')
    plt.title('swiergotliwy z zadania 3 zaszumiony (SNR = {} dB)'.format(snr_db))
    plt.legend()
    plt.grid(True)

    plt.subplot(3,1,2)
    plt.plot(t, oryg, label='czysty sygnal')
    plt.xlabel('czas')
    plt.ylabel('Amplituda')
    plt.title('swiergotliwy z zadania 3 niezaszumiony '.format(snr_db))
    plt.legend()
    plt.grid(True)

    plt.subplot(3,1,3)
    plt.plot(t, dsignal, label='odszumiony sygnal')
    plt.xlabel('czas')
    plt.ylabel('Amplituda')
    plt.title('swiergotliwy z zadania 3 odszumianie za pomoca wienera (SNR = {} dB)'.format(snr_db))
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def zad5(szum_syg,t, snr_db,oryg):

    window_length = 5
    polyorder = 2
    d = sig.savgol_filter(szum_syg, window_length, polyorder)

    plt.figure(figsize=(10, 6))

    plt.subplot(3,1,1)
    plt.plot(t, szum_syg, label='zaszumiony sygnal')
    plt.xlabel('czas')
    plt.ylabel('Amplituda')
    plt.title('swiergotliwy z zadania 3 zaszumiony (SNR = {} dB)'.format(snr_db))
    plt.legend()
    plt.grid(True)

    plt.subplot(3,1,2)
    plt.plot(t, oryg, label='czysty sygnal')
    plt.xlabel('czas')
    plt.ylabel('Amplituda')
    plt.title('swiergotliwy z zadania 3 niezaszumiony '.format(snr_db))
    plt.legend()
    plt.grid(True)

    plt.subplot(3,1,3)
    plt.plot(t, d, label='odszumiony sygnal')
    plt.xlabel('czas')
    plt.ylabel('Amplituda')
    plt.title('swiergotliwy z zadania 3 odszumianie za pomoca savitzky-golay (SNR = {} dB)'.format(snr_db))
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def zad6(szum_syg,t, snr_db,oryg):
    emd = EMD()
    IMFs = emd(szum_syg)

    #zmienna
    kept_IMFs = range(min(5, len(IMFs)))

    syngal_rek = np.sum(IMFs[kept_IMFs], axis=0)

    plt.figure(figsize=(10, 6))

    plt.subplot(3,1,1)
    plt.plot(t, szum_syg, label='zaszumiony sygnal')
    plt.xlabel('czas')
    plt.ylabel('Amplituda')
    plt.title('swiergotliwy z zadania 3 zaszumiony (SNR = {} dB)'.format(snr_db))
    plt.legend()
    plt.grid(True)

    plt.subplot(3,1,2)
    plt.plot(t, oryg, label='czysty sygnal')
    plt.xlabel('czas')
    plt.ylabel('Amplituda')
    plt.title('swiergotliwy z zadania 3 niezaszumiony '.format(snr_db))
    plt.legend()
    plt.grid(True)

    plt.subplot(3,1,3)
    plt.plot(t, syngal_rek, label='odszumiony sygnal')
    plt.xlabel('czas')
    plt.ylabel('Amplituda')
    plt.title('swiergotliwy z zadania 3 odszumianie za pomoca emd i rekonstrukcji (SNR = {} dB)'.format(snr_db))
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def z7():
    df_signal = pd.read_csv('signal4.csv')
    df_ns = pd.read_csv('ns.csv')

    t = df_signal['Time'].values[:100]
    signal = df_signal['Signal'].values[:100]
    szum_syg = df_ns['NoisySignal'].values[:100]

    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(t, szum_syg, label='zaszumiony sygnal')
    plt.xlabel('czas')
    plt.ylabel('Amplituda')
    plt.title('swiergotliwy z szumami')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(t, signal, label='czysty sygnal')
    plt.xlabel('czas')
    plt.ylabel('Amplituda')
    plt.title('swiergotliwy')
    plt.legend()
    plt.grid(True)


    plt.tight_layout()
    plt.show()

    zad4(szum_syg, t, 5, signal)
    zad5(szum_syg, t, 5, signal)
    zad6(szum_syg, t, 5, signal)


z1()
z2()
z3()
z7()

