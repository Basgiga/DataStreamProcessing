import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
import pywt
import scipy.signal as sig
import matplotlib.pyplot as plt
import numpy as np
import time
import math
import pandas as pd



def zad1():
    def plot_haar(level):
        fa_haar = pywt.Wavelet('haar')
        phi_haar, psi_haar, x_haar = fa_haar.wavefun(level=level)
        plt.plot(x_haar, psi_haar)
        plt.title('Haar Wavelet')
        plt.xlabel('x')
        plt.ylabel('Amplitude')
        plt.grid(True)
        plt.show()

    def plot_db1(level):
        fa_db1 = pywt.Wavelet('db2')
        phi_db1, psi_db1, x_db1 = fa_db1.wavefun(level=level)
        plt.plot(x_db1, psi_db1)
        plt.title('Daubechies Wavelet (db2)')
        plt.xlabel('x')
        plt.ylabel('Amplitude')
        plt.grid(True)
        plt.show()

    def plot_sym2(level):
        fa_sym2 = pywt.Wavelet('sym5')
        phi_sym2, psi_sym2, x_sym2 = fa_sym2.wavefun(level=level)
        plt.plot(x_sym2, psi_sym2)
        plt.title('Symlets Wavelet (sym2)')
        plt.xlabel('x')
        plt.ylabel('Amplitude')
        plt.grid(True)
        plt.show()

    def plot_coif2(level):
        fa_coif2 = pywt.Wavelet('coif2')
        phi_coif2, psi_coif2, x_coif2 = fa_coif2.wavefun(level=level)
        plt.plot(x_coif2, psi_coif2)
        plt.title('Coiflets Wavelet (coif2)')
        plt.xlabel('x')
        plt.ylabel('Amplitude')
        plt.grid(True)
        plt.show()

    def plot_bior35(level):
        fa_bior35 = pywt.Wavelet('bior3.5')
        phi_d, psi_d, phi_r, psi_r, x = fa_bior35.wavefun(level=level)
        plt.plot(psi_d)
        plt.plot(psi_r)
        plt.title('Biorthogonal Wavelet (bior3.5)')
        plt.xlabel('x')
        plt.ylabel('Amplitude')
        plt.grid(True)
        plt.show()

    def plot_gaus1(level):
        fa_gaus1 = pywt.ContinuousWavelet('gaus1')
        phi_gaus1, psi_gaus1 = fa_gaus1.wavefun(level=level)
        plt.plot(phi_gaus1)
        plt.title('Gaussian Wavelet (gaus1)')
        plt.xlabel('x')
        plt.ylabel('Amplitude')
        plt.grid(True)
        plt.show()

    def plot_mexh(level):
        fa_mexh = pywt.ContinuousWavelet('mexh')
        phi_mexh, psi_mexh = fa_mexh.wavefun(level=level)
        plt.plot(phi_mexh)
        plt.title('Mexican Hat Wavelet (mexh)')
        plt.xlabel('x')
        plt.ylabel('Amplitude')
        plt.grid(True)
        plt.show()

    def plot_morl(level):
        fa_morl = pywt.ContinuousWavelet('morl')
        phi_morl, psi_morl = fa_morl.wavefun(level=level)
        plt.plot(phi_morl)
        plt.title('Morlet Wavelet (morl)')
        plt.xlabel('x')
        plt.ylabel('Amplitude')
        plt.grid(True)
        plt.show()

    level = 8

    root = tk.Tk()
    root.title("falki :D")

    button_haar = tk.Button(root, text="Haar", command=lambda : plot_haar(poziom_slider.get()))
    button_haar.pack()

    button_db1 = tk.Button(root, text="Daubechies (db2)", command=lambda :plot_db1(poziom_slider.get()))
    button_db1.pack()

    button_sym2 = tk.Button(root, text="Symlets (sym5)", command=lambda :plot_sym2(poziom_slider.get()))
    button_sym2.pack()

    button_coif2 = tk.Button(root, text="Coiflets (coif2)", command=lambda :plot_coif2(poziom_slider.get()))
    button_coif2.pack()

    button_bior35 = tk.Button(root, text="Biorthogonal (bior3.5)", command=lambda :plot_bior35(poziom_slider.get()))
    button_bior35.pack()

    button_gaus1 = tk.Button(root, text="Gaussian (gaus1)", command=lambda :plot_gaus1(poziom_slider.get()))
    button_gaus1.pack()

    button_mexh = tk.Button(root, text="Mexican Hat (mexh)", command=lambda :plot_mexh(poziom_slider.get()))
    button_mexh.pack()

    button_morl = tk.Button(root, text="Morlet (morl)", command=lambda :plot_morl(poziom_slider.get()))
    button_morl.pack()


    label_t_trwania2 = tk.Label(root, text='zmienna level:')
    label_t_trwania2.pack()
    poziom_slider = tk.Scale(root, from_=5.0, to=10.0, orient=tk.HORIZONTAL, resolution=1.0)
    poziom_slider.pack()


    root.mainloop()

def zad2():
    def plot_haar(v,level):
        falka = f'haar{v}'
        fa_haar = pywt.Wavelet(falka)
        phi_haar, psi_haar, x_haar = fa_haar.wavefun(level=level)
        plt.plot(x_haar, psi_haar)
        plt.title('Haar Wavelet')
        plt.xlabel('x')
        plt.ylabel('Amplitude')
        plt.grid(True)
        plt.show()

    def plot_db1(v,level):
        falka = f'db{v}'
        fa_db1 = pywt.Wavelet(falka)
        phi_db1, psi_db1, x_db1 = fa_db1.wavefun(level=level)
        plt.plot(x_db1, psi_db1)
        plt.title(f'Daubechies Wavelet (db{v})')
        plt.xlabel('x')
        plt.ylabel('Amplitude')
        plt.grid(True)
        plt.show()

    def plot_sym2(v,level):
        falka = f'sym{v}'
        fa_sym2 = pywt.Wavelet(falka)
        phi_sym2, psi_sym2, x_sym2 = fa_sym2.wavefun(level=level)
        plt.plot(x_sym2, psi_sym2)
        plt.title(f'Symlets Wavelet (sym{v})')
        plt.xlabel('x')
        plt.ylabel('Amplitude')
        plt.grid(True)
        plt.show()

    def plot_coif2(v,level):
        falka = f'coif{v}'
        fa_coif2 = pywt.Wavelet(falka)
        phi_coif2, psi_coif2, x_coif2 = fa_coif2.wavefun(level=level)
        plt.plot(x_coif2, psi_coif2)
        plt.title(f'Coiflets falka (coif{v})')
        plt.xlabel('x')
        plt.ylabel('Amplitude')
        plt.grid(True)
        plt.show()

    def plot_bior35(v,level):
        falka = f'bior{v}'
        fa_bior35 = pywt.Wavelet(falka)
        phi_d, psi_d, phi_r, psi_r, x = fa_bior35.wavefun(level=level)
        plt.plot(psi_d)
        plt.plot(psi_r)
        plt.title(f'Biorthogonal falka (bior{v})')
        plt.xlabel('x')
        plt.ylabel('Amplitude')
        plt.grid(True)
        plt.show()

    def plot_gaus1(v,level):
        falka = f'gaus{v}'
        fa_gaus1 = pywt.ContinuousWavelet(falka)
        phi_gaus1, psi_gaus1 = fa_gaus1.wavefun(level=level)
        plt.plot(phi_gaus1)
        plt.title(f'Gaussian Wavelet (gaus{v})')
        plt.xlabel('x')
        plt.ylabel('Amplitude')
        plt.grid(True)
        plt.show()

    def plot_mexh(v,level):
        falka = f'mexh'
        fa_mexh = pywt.ContinuousWavelet(falka)
        phi_mexh, psi_mexh = fa_mexh.wavefun(level=level)
        plt.plot(phi_mexh)
        plt.title('Mexican Hat Wavelet (mexh)')
        plt.xlabel('x')
        plt.ylabel('Amplitude')
        plt.grid(True)
        plt.show()

    def plot_morl(level):
        falka = f'morl'
        fa_morl = pywt.ContinuousWavelet(falka)
        phi_morl, psi_morl = fa_morl.wavefun(level=level)
        plt.plot(phi_morl)
        plt.title('Morlet Wavelet (morl)')
        plt.xlabel('x')
        plt.ylabel('Amplitude')
        plt.grid(True)
        plt.show()




    def plot_daubechies_wavelet(v, level):
        falka = f'db{v}'
        fa = pywt.Wavelet(falka)
        phi, psi, x = fa.wavefun(level=level)

        plt.figure(figsize=(10, 4))
        plt.plot(x, psi)
        plt.title(f'Daubechies (wersja: {v}, Level: {level})')
        plt.xlabel('x')
        plt.ylabel('Amplitude')
        plt.grid(True)
        plt.show()

    root = tk.Tk()
    root.title("Daubechies falki roznej wersji")


    version_label = ttk.Label(root, text="ktora werjsa:")
    version_label.pack()
    version_entry = ttk.Entry(root)
    version_entry.pack()
    version_entry.insert(0, '1')

    label_t_trwania2 = tk.Label(root, text='zmienna level:')
    label_t_trwania2.pack()
    poziom_slider = tk.Scale(root, from_=5.0, to=10.0, orient=tk.HORIZONTAL, resolution=1.0)
    poziom_slider.pack()


    plot_button = ttk.Button(root, text="db", command= lambda: plot_daubechies_wavelet(version_entry.get(), poziom_slider.get()))
    plot_button.pack()

    plot_button2 = ttk.Button(root, text="haar",
                             command=lambda: plot_haar(version_entry.get(), poziom_slider.get()))
    plot_button2.pack()

    plot_button3 = ttk.Button(root, text="sym",
                             command=lambda: plot_sym2(version_entry.get(), poziom_slider.get()))
    plot_button3.pack()

    plot_button4 = ttk.Button(root, text="coif",
                             command=lambda: plot_coif2(version_entry.get(), poziom_slider.get()))
    plot_button4.pack()

    plot_button5 = ttk.Button(root, text="bior",
                             command=lambda: plot_bior35(version_entry.get(), poziom_slider.get()))
    plot_button5.pack()

    plot_button6 = ttk.Button(root, text="gaus",
                             command=lambda: plot_gaus1(version_entry.get(), poziom_slider.get()))
    plot_button6.pack()

    plot_button7 = ttk.Button(root, text="Plot",
                             command=lambda: plot_mexh(version_entry.get(), poziom_slider.get()))
    plot_button7.pack()
    root.mainloop()


def zad3():
    def plot_haar(level, amplituda, frekwencja, czest_prob, t_trwania):
        t = np.linspace(0, t_trwania, int(czest_prob * t_trwania), endpoint=False)
        swiergot = amplituda * sig.chirp(2 * np.pi * frekwencja * t, 6, 10, 1)

        haar_coeffs = pywt.wavedec(swiergot, 'haar')


        plt.figure(figsize=(12, 8))

        plt.subplot(2, 1, 1)
        plt.plot(t, swiergot)
        plt.title('wygenerowany sygnal')

        plt.subplot(2, 1, 2)
        for i in range(len(haar_coeffs)):
            plt.plot(haar_coeffs[i], label=f'Level {i + 1}')
        plt.title('dekompozycja (falka Haar)')
        plt.legend()

        plt.show()

    def plot_db1(level, amplituda, frekwencja, czest_prob, t_trwania):
        t = np.linspace(0, t_trwania, int(czest_prob * t_trwania), endpoint=False)
        swiergot = amplituda * sig.chirp(2 * np.pi * frekwencja * t, 6, 10, 1)

        haar_coeffs = pywt.wavedec(swiergot, 'db2')

        plt.figure(figsize=(12, 8))

        plt.subplot(2, 1, 1)
        plt.plot(t, swiergot)
        plt.title('wygenerowany sygnal')

        plt.subplot(2, 1, 2)
        for i in range(len(haar_coeffs)):
            plt.plot(haar_coeffs[i], label=f'Level {i + 1}')
        plt.title('dekompozycja (falka Daubechies)')
        plt.legend()

        plt.show()

    def plot_sym2(level, amplituda, frekwencja, czest_prob, t_trwania):
        t = np.linspace(0, t_trwania, int(czest_prob * t_trwania), endpoint=False)
        swiergot = amplituda * sig.chirp(2 * np.pi * frekwencja * t, 6, 10, 1)

        haar_coeffs = pywt.wavedec(swiergot, 'sym5')

        plt.figure(figsize=(12, 8))

        plt.subplot(2, 1, 1)
        plt.plot(t, swiergot)
        plt.title('wygenerowany sygnal')

        plt.subplot(2, 1, 2)
        for i in range(len(haar_coeffs)):
            plt.plot(haar_coeffs[i], label=f'Level {i + 1}')
        plt.title('dekompozycja (falka Symlets)')
        plt.legend()

        plt.show()

    def plot_coif2(level, amplituda, frekwencja, czest_prob, t_trwania):
        t = np.linspace(0, t_trwania, int(czest_prob * t_trwania), endpoint=False)
        swiergot = amplituda * sig.chirp(2 * np.pi * frekwencja * t, 6, 10, 1)

        haar_coeffs = pywt.wavedec(swiergot, 'coif2')

        plt.figure(figsize=(12, 8))

        plt.subplot(2, 1, 1)
        plt.plot(t, swiergot)
        plt.title('wygenerowany sygnal')

        plt.subplot(2, 1, 2)
        for i in range(len(haar_coeffs)):
            plt.plot(haar_coeffs[i], label=f'Level {i + 1}')
        plt.title('dekompozycja (falka Coiflets)')
        plt.legend()

        plt.show()

    def plot_bior35(level, amplituda, frekwencja, czest_prob, t_trwania):
        t = np.linspace(0, t_trwania, int(czest_prob * t_trwania), endpoint=False)
        swiergot = amplituda * sig.chirp(2 * np.pi * frekwencja * t, 6, 10, 1)

        haar_coeffs = pywt.wavedec(swiergot, 'bior3.5')

        plt.figure(figsize=(12, 8))

        plt.subplot(2, 1, 1)
        plt.plot(t, swiergot)
        plt.title('wygenerowany sygnal')

        plt.subplot(2, 1, 2)
        for i in range(len(haar_coeffs)):
            plt.plot(haar_coeffs[i], label=f'Level {i + 1}')
        plt.title('dekompozycja (falka Biortogonalna)')
        plt.legend()

        plt.show()

    def plot_gaus1(level, amplituda, frekwencja, czest_prob, t_trwania):
        t = np.linspace(0, t_trwania, int(czest_prob * t_trwania), endpoint=False)
        swiergot = amplituda * sig.chirp(2 * np.pi * frekwencja * t, 6, 10, 1)

        scales = np.arange(1, level + 1)  # Zakres skali
        coef, freqs = pywt.cwt(swiergot, scales, 'gaus1')

        plt.figure(figsize=(12, 8))

        plt.subplot(2, 1, 1)
        plt.plot(t, swiergot)
        plt.title('wygenerowany sygnal')

        plt.subplot(2, 1, 2)
        for i in range(len(coef)):
            plt.plot(coef[i], label=f'Level {i + 1}')
        plt.title('dekompozycja (falka Gaussowska)')
        plt.legend()

        plt.show()

    def plot_mexh(level, amplituda, frekwencja, czest_prob, t_trwania):
        t = np.linspace(0, t_trwania, int(czest_prob * t_trwania), endpoint=False)
        swiergot = amplituda * sig.chirp(2 * np.pi * frekwencja * t, 6, 10, 1)

        scales = np.arange(1, level + 1)  # Zakres skali
        coef, freqs = pywt.cwt(swiergot, scales, 'mexh')

        plt.figure(figsize=(12, 8))

        plt.subplot(2, 1, 1)
        plt.plot(t, swiergot)
        plt.title('wygenerowany sygnal')

        plt.subplot(2, 1, 2)
        for i in range(len(coef)):
            plt.plot(coef[i], label=f'Level {i + 1}')
        plt.title('dekompozycja (falka Gaussowska)')
        plt.legend()

        plt.show()

    def plot_morl(level, amplituda, frekwencja, czest_prob, t_trwania):
        t = np.linspace(0, t_trwania, int(czest_prob * t_trwania), endpoint=False)
        swiergot = amplituda * sig.chirp(2 * np.pi * frekwencja * t, 6, 10, 1)

        scales = np.arange(1, level + 1)  # Zakres skali
        coef, freqs = pywt.cwt(swiergot, scales, 'morl')

        plt.figure(figsize=(12, 8))

        plt.subplot(2, 1, 1)
        plt.plot(t, swiergot)
        plt.title('wygenerowany sygnal')

        plt.subplot(2, 1, 2)
        for i in range(len(coef)):
            plt.plot(coef[i], label=f'Level {i + 1}')
        plt.title('dekompozycja (falka Gaussowska)')
        plt.legend()

        plt.show()



    root = tk.Tk()
    root.title("falki :D")


    label_t_trwania2 = tk.Label(root, text='zmienna level:')
    label_t_trwania2.pack()
    poziom_slider = tk.Scale(root, from_=5.0, to=10.0, orient=tk.HORIZONTAL, resolution=1.0)
    poziom_slider.pack()

    label_amplituda = tk.Label(root, text='Amplituda:')
    label_amplituda.pack()
    slider_amplituda = tk.Scale(root, from_=0.1, to=2.0, orient=tk.HORIZONTAL, resolution=0.1)
    slider_amplituda.pack()

    label_frekwencja = tk.Label(root, text='Częstotliwość:')
    label_frekwencja.pack()
    slider_frekwencja = tk.Scale(root, from_=1.0, to=10.0, orient=tk.HORIZONTAL, resolution=0.5)
    slider_frekwencja.pack()

    label_czest_prob = tk.Label(root, text='Częstotliwość próbkowania:')
    label_czest_prob.pack()
    slider_czest_prob = tk.Scale(root, from_=100, to=2000, orient=tk.HORIZONTAL, resolution=100)
    slider_czest_prob.pack()

    label_t_trwania = tk.Label(root, text='Czas trwania:')
    label_t_trwania.pack()
    slider_t_trwania = tk.Scale(root, from_=0.5, to=2.0, orient=tk.HORIZONTAL, resolution=0.1)
    slider_t_trwania.pack()


    button_haar = tk.Button(root, text="Haar", command=lambda: plot_haar(poziom_slider.get(), slider_amplituda.get(), slider_frekwencja.get(), slider_czest_prob.get(), slider_t_trwania.get()))
    button_haar.pack()

    button_db1 = tk.Button(root, text="Daubechies (db2)", command=lambda: plot_db1(poziom_slider.get(), slider_amplituda.get(), slider_frekwencja.get(), slider_czest_prob.get(), slider_t_trwania.get()))
    button_db1.pack()

    button_sym2 = tk.Button(root, text="Symlets (sym5)", command=lambda: plot_sym2(poziom_slider.get(), slider_amplituda.get(), slider_frekwencja.get(), slider_czest_prob.get(), slider_t_trwania.get()))
    button_sym2.pack()

    button_coif2 = tk.Button(root, text="Coiflets (coif2)", command=lambda: plot_coif2(poziom_slider.get(), slider_amplituda.get(), slider_frekwencja.get(), slider_czest_prob.get(), slider_t_trwania.get()))
    button_coif2.pack()

    button_bior35 = tk.Button(root, text="Biorthogonal (bior3.5)", command=lambda: plot_bior35(poziom_slider.get(), slider_amplituda.get(), slider_frekwencja.get(), slider_czest_prob.get(), slider_t_trwania.get()))
    button_bior35.pack()

    button_gaus1 = tk.Button(root, text="Gaussian (gaus1)", command=lambda: plot_gaus1(poziom_slider.get(), slider_amplituda.get(), slider_frekwencja.get(), slider_czest_prob.get(), slider_t_trwania.get()))
    button_gaus1.pack()

    button_mexh = tk.Button(root, text="Mexican Hat (mexh)", command=lambda: plot_mexh(poziom_slider.get(), slider_amplituda.get(), slider_frekwencja.get(), slider_czest_prob.get(), slider_t_trwania.get()))
    button_mexh.pack()

    button_morl = tk.Button(root, text="Morlet (morl)", command=lambda: plot_morl(poziom_slider.get(), slider_amplituda.get(), slider_frekwencja.get(), slider_czest_prob.get(), slider_t_trwania.get()))
    button_morl.pack()

    root.mainloop()


#zad1()
zad2()
zad3()

# zadanie 1 wyslac