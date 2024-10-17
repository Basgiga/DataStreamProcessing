import scipy.signal as sig
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import numpy as np
import time
import math
import pandas as pd
import tkinter as tk
from tkinter import ttk
from scipy.signal import windows, freqz
from scipy.fft import fft, fftshift

def zad1(N, fs):
    # Okna
    hamming_window = windows.hamming(N)
    hann_window = windows.hann(N)
    blackman_window = windows.blackman(N)
    dirichlet_window = windows.boxcar(N)

    # Obliczanie widm amplitudowych
    def compute_fft(window):
        return np.abs(np.fft.fft(window, 2048))

    hamming_fft = compute_fft(hamming_window)
    hann_fft = compute_fft(hann_window)
    blackman_fft = compute_fft(blackman_window)
    dirichlet_fft = compute_fft(dirichlet_window)

    freqs = np.fft.fftfreq(2048, d=1 / fs)

    plt.figure(figsize=(12, 16))

    # okna (w czasie)
    plt.subplot(4, 2, 1)
    plt.plot(hamming_window, color='b', linestyle='-', marker='o')
    plt.title('okno Hamminga')
    plt.ylabel('Amplituda')
    plt.grid(True)

    plt.subplot(4, 2, 2)
    plt.plot(hann_window, color='g', linestyle='--', marker='s')
    plt.title('okno Hanna')
    plt.ylabel('Amplituda')
    plt.grid(True)

    plt.subplot(4, 2, 3)
    plt.plot(blackman_window, color='r', linestyle='-.', marker='^')
    plt.title('okno Blackmana')
    plt.ylabel('Amplituda')
    plt.grid(True)

    plt.subplot(4, 2, 4)
    plt.plot(dirichlet_window, color='m', linestyle=':', marker='x')
    plt.title('okno Dirichleta')
    plt.ylabel('Amplituda')
    plt.grid(True)

    # widma
    plt.subplot(4, 2, 5)
    plt.plot(freqs[:len(hamming_fft)], 20 * np.log10(hamming_fft), color='b', label='Hamming')
    plt.title('widmo amplitudowe - Hamming')
    plt.ylabel('Amplituda [dB]')
    plt.grid(True)
    plt.legend()

    plt.subplot(4, 2, 6)
    plt.plot(freqs[:len(hann_fft)], 20 * np.log10(hann_fft), color='g', label='Hann')
    plt.title('widmo amplitudowe - Hanna')
    plt.ylabel('Amplituda [dB]')
    plt.grid(True)
    plt.legend()

    plt.subplot(4, 2, 7)
    plt.plot(freqs[:len(blackman_fft)], 20 * np.log10(blackman_fft), color='r', label='Blackman')
    plt.title('widmo amplitudowe - Blackmana')
    plt.xlabel('frekwencje [Hz]')
    plt.ylabel('Amplituda [dB]')
    plt.grid(True)
    plt.legend()

    plt.subplot(4, 2, 8)
    plt.plot(freqs[:len(dirichlet_fft)], 20 * np.log10(dirichlet_fft), color='m', label='Dirichlet')
    plt.title('widmo amplitudowe - Dirichleta')
    plt.xlabel('Frekwencje [Hz]')
    plt.ylabel('Amplituda [dB]')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

def zad2(N, fs, f1, f2, f3, amp, t_trwania):
    t = np.linspace(0, t_trwania, int(fs * t_trwania), endpoint=False)

    sin1 = amp * np.sin(2 * np.pi * f1 * t)
    sin2 = amp * np.sin(2 * np.pi * f2 * t)
    sin3 = amp * np.sin(2 * np.pi * f3 * t)

    sygnaly = [sin1, sin2, sin3]
    frekwencje = [f1, f2, f3]

    hamming_window = windows.hamming(len(t))
    hann_window = windows.hann(len(t))
    blackman_window = windows.blackman(len(t))
    dirichlet_window = windows.boxcar(len(t))

    # Obliczanie widm amplitudowych
    def compute_fft(sygnal, okno):
        sygnal_okno = sygnal * okno
        return np.abs(np.fft.fft(sygnal_okno, 2048))

    hamming_ffts = [compute_fft(sygnal, hamming_window) for sygnal in sygnaly]
    hann_ffts = [compute_fft(sygnal, hann_window) for sygnal in sygnaly]
    blackman_ffts = [compute_fft(sygnal, blackman_window) for sygnal in sygnaly]
    dirichlet_ffts = [compute_fft(sygnal, dirichlet_window) for sygnal in sygnaly]


    plt.figure(figsize=(12, 8))

    # Wykresy dla okna Hamminga
    plt.subplot(3, 3, 1)
    for fft, f in zip(hamming_ffts, frekwencje):
        plt.plot(np.fft.fftfreq(len(fft))* fs, fft, label=f'{f} Hz')
    plt.title('Widma amplitudowe - okno Hamminga')
    plt.xlabel('Częstotliwość [Hz]')
    plt.ylabel('Amplituda')
    plt.legend()

    # Wykresy dla okna Hanninga
    plt.subplot(3, 3, 2)
    for fft, f in zip(hann_ffts, frekwencje):
        plt.plot(np.fft.fftfreq(len(fft))* fs, fft, label=f'{f} Hz')
    plt.title('Widma amplitudowe - okno Hanninga')
    plt.xlabel('Częstotliwość [Hz]')
    plt.ylabel('Amplituda')
    plt.legend()

    # Wykresy dla okna Blackmana
    plt.subplot(3, 3, 3)
    for fft, f in zip(blackman_ffts, frekwencje):
        plt.plot(np.fft.fftfreq(len(fft))* fs, fft, label=f'{f} Hz')
    plt.title('Widma amplitudowe - okno Blackmana')
    plt.xlabel('Częstotliwość [Hz]')
    plt.ylabel('Amplituda')
    plt.legend()

    # Wykresy dla okna Dirichleta (prostokątnego)
    plt.subplot(3, 3, 4)
    for fft, f in zip(dirichlet_ffts, frekwencje):
        plt.plot(np.fft.fftfreq(len(fft)) * fs, fft, label=f'{f} Hz')
    plt.title('Widma amplitudowe - okno Dirichleta')
    plt.xlabel('Częstotliwość [Hz]')
    plt.ylabel('Amplituda')
    plt.legend()

    # wykresy sinosów

    plt.subplot(3, 3, 7)
    plt.plot(t, sin1, label=f'Sygnał {f1} Hz')
    plt.title(f'Sinus z {f1} Hz')
    plt.xlabel('Czas [s]')
    plt.ylabel('Amplituda')
    plt.legend()

    plt.subplot(3, 3, 8)
    plt.plot(t, sin2, label=f'Sygnał {f2} Hz')
    plt.title(f'Sinus z {f2} Hz')
    plt.xlabel('Czas [s]')
    plt.ylabel('Amplituda')
    plt.legend()

    plt.subplot(3, 3, 9)
    plt.plot(t, sin3, label=f'Sygnał {f3} Hz')
    plt.title(f'Sinus z {f3} Hz')
    plt.xlabel('Czas [s]')
    plt.ylabel('Amplituda')
    plt.legend()

    plt.tight_layout()
    plt.show()


def zad3(N, fs, f1, f2, f3, amp, t_trwania):
    t = np.linspace(0, t_trwania, int(fs * t_trwania), endpoint=False)

    sin1 = amp * np.sin(2 * np.pi * f1 * t)
    sin2 = amp * np.sin(2 * np.pi * f2 * t)
    sin3 = amp * np.sin(2 * np.pi * f3 * t)

    sygnaly = [sin1, sin2, sin3]
    frekwencje = [f1, f2, f3]

    # Wykonanie FFT
    widma =[fft(sygnal) for sygnal in sygnaly ]

    # Wyliczenie czest
    frequencje_fft= [np.fft.fftfreq(len(widmo)) * fs for widmo in widma]

    # Wykresy widm amplitudowych
    plt.figure(figsize=(12, 8))

    for i, (widmo, f) in enumerate(zip(widma, frekwencje), start=1):
        plt.subplot(3, 1, i)
        plt.plot(frequencje_fft[i - 1], np.abs(widmo))
        plt.title(f'Widmo amplitudowe - Sygnał {f} Hz')
        plt.xlabel('Częstotliwość [Hz]')
        plt.ylabel('Amplituda')
        plt.grid(True)

    plt.tight_layout()
    plt.show()

def z1():
    root = tk.Tk()
    root.title("Manipulacja parametrami")

    label_czest_prob = tk.Label(root, text='Częstotliwość próbkowania:')
    label_czest_prob.grid(row=2, column=0)
    slider_czest_prob = tk.Scale(root, from_=100, to=2000, orient=tk.HORIZONTAL, resolution=100)
    slider_czest_prob.grid(row=2, column=1)

    label_t_trwania = tk.Label(root, text='Dlugosc okna:')
    label_t_trwania.grid(row=3, column=0)
    slider_t_trwania = tk.Scale(root, from_=10, to=200, orient=tk.HORIZONTAL, resolution=10)
    slider_t_trwania.grid(row=3, column=1)


    button_generate5 = ttk.Button(root, text="zad1", command=lambda: zad1(slider_t_trwania.get(), slider_czest_prob.get()))
    button_generate5.grid(row=4, columnspan=2)

    label_amplituda = tk.Label(root, text='Amplituda:')
    label_amplituda.grid(row=5, column=0)
    slider_amplituda = tk.Scale(root, from_=1.0, to=2.0, orient=tk.HORIZONTAL, resolution=0.1)
    slider_amplituda.grid(row=5, column=1)

    label_frekwencja1 = tk.Label(root, text='Częstotliwość 1:')
    label_frekwencja1.grid(row=6, column=0)
    slider_frekwencja1 = tk.Scale(root, from_=10.0, to=20.0, orient=tk.HORIZONTAL, resolution=1.0)
    slider_frekwencja1.grid(row=6, column=1)
    label_frekwencja2 = tk.Label(root, text='Częstotliwość 2:')
    label_frekwencja2.grid(row=7, column=0)
    slider_frekwencja2 = tk.Scale(root, from_=10.0, to=30.0, orient=tk.HORIZONTAL, resolution=1.0)
    slider_frekwencja2.grid(row=7, column=1)
    label_frekwencja3 = tk.Label(root, text='Częstotliwość 3:')
    label_frekwencja3.grid(row=8, column=0)
    slider_frekwencja3 = tk.Scale(root, from_=10.0, to=40.0, orient=tk.HORIZONTAL, resolution=1.0)
    slider_frekwencja3.grid(row=8, column=1)

    label_czest_prob2 = tk.Label(root, text='Częstotliwość próbkowania:')
    label_czest_prob2.grid(row=9, column=0)
    slider_czest_prob2 = tk.Scale(root, from_=100, to=2000, orient=tk.HORIZONTAL, resolution=100)
    slider_czest_prob2.grid(row=9, column=1)

    label_t_trwania2 = tk.Label(root, text='Czas trwania:')
    label_t_trwania2.grid(row=10, column=0)
    slider_t_trwania2 = tk.Scale(root, from_=1.0, to=2.0, orient=tk.HORIZONTAL, resolution=0.1)
    slider_t_trwania2.grid(row=10, column=1)

    button_generate2 = ttk.Button(root, text="zad2",
                                  command=lambda: zad2(slider_t_trwania.get(), slider_czest_prob2.get(),  slider_frekwencja1.get(), slider_frekwencja2.get(), slider_frekwencja3.get(), slider_amplituda.get(), slider_t_trwania2.get()))
    button_generate2.grid(row=11, columnspan=2)

    button_generate3 = ttk.Button(root, text="zad3",
                                  command=lambda: zad3(slider_t_trwania.get(), slider_czest_prob2.get(),
                                                       slider_frekwencja1.get(), slider_frekwencja2.get(),
                                                       slider_frekwencja3.get(), slider_amplituda.get(),
                                                       slider_t_trwania2.get()))
    button_generate3.grid(row=12, columnspan=2)


    root.mainloop()
def zad4(plik):
    data = pd.read_csv(plik, sep=';')

    czas = data['time']
    wartosci_sygnalu = data['signal1']

    # fs (zakladajac staly krok czasowy)
    fs = 1 / (czas[1] - czas[0])

    # Obliczenie FFT dla sygnału
    fft_result = np.fft.fft(wartosci_sygnalu)

    # Obliczenie frq odpowiadających punktom FFT
    freqs = np.fft.fftfreq(len(data), d=1 / fs)

    # polowa (lustro)
    half_length = len(data) // 2
    fft_result = fft_result[:half_length]
    freqs = freqs[:half_length]

    plt.figure(figsize=(12, 8))

    # Wykres sygnału
    plt.subplot(1, 2, 1)
    plt.plot(czas, wartosci_sygnalu)
    plt.title('Sygnał')
    plt.xlabel('Czas [s]')
    plt.ylabel('Wartość')
    plt.grid(True)

    # Wykres widma
    plt.subplot(1, 2, 2)
    plt.plot(freqs, np.abs(fft_result))
    plt.title('Widmo sygnału')
    plt.xlabel('Częstotliwość [Hz]')
    plt.ylabel('Amplituda')
    plt.grid(True)

    plt.tight_layout()
    plt.show()



def z4():
    root = tk.Tk()
    root.title("Manipulacja parametrami")

    version_label = ttk.Label(root, text="podaj plik")
    version_label.pack()
    version_entry = ttk.Entry(root)
    version_entry.pack()
    version_entry.insert(0, '1')

    button_generate3 = ttk.Button(root, text="zad4",
                                  command=lambda: zad4(str(version_entry.get())))
    button_generate3.pack()

    root.mainloop()
#z1()
z4()