import scipy.signal as sig
import matplotlib.pyplot as plt
import numpy as np
import time
import math
import pandas as pd
import tkinter as tk
from tkinter import ttk
from scipy.io import wavfile
import librosa
import librosa.display
from matplotlib.animation import FuncAnimation
from PyEMD import EMD


def zad1(amplituda, frekwencja, czest_prob, t_trwania):

    t = np.linspace(0, t_trwania, int(czest_prob * t_trwania), endpoint=False)

    # Tworzenie sygnałów

    #sinus ewentualny:
    sinus = amplituda * sig.chirp(t, f0=frekwencja, f1=frekwencja, t1=1, method='linear', phi=-90)  # Ustawienie t1=1 daje stałą częstotliwość

    #sinus = amplituda * np.sin(2 * np.pi * frekwencja * t)

    prosto = amplituda * sig.square(2 * np.pi * frekwencja * t)

    pila = amplituda * sig.sawtooth(2 * np.pi * frekwencja * t)

    swiergot = amplituda * sig.chirp(2 * np.pi * frekwencja * t, 6, 10, 1)

    superpozycja_sin_cos = amplituda * sig.chirp(t, f0=frekwencja, f1=frekwencja, t1=1, method='linear', phi=-90) + amplituda * sig.chirp(t, f0=frekwencja, f1=frekwencja, t1=1, method='linear', phi=0)

    #dirac = np.zeros_like(t)
    #dirac[0] = 1
    dirac = amplituda * sig.unit_impulse(int(t_trwania*czest_prob))
    dirac = dirac + 0.01

    #tworzenie subplotow
    plt.figure(figsize=(10, 8))

    plt.subplot(6, 1, 1)
    plt.plot(t, sinus)
    plt.title('Sinus')
    plt.xlabel('Czas')
    plt.ylabel('Amplituda')
    plt.grid(True)

    plt.subplot(6, 1, 2)
    plt.plot(t, prosto)
    plt.title('Prostokąt')
    plt.xlabel('Czas')
    plt.ylabel('Amplituda')
    plt.grid(True)

    plt.subplot(6, 1, 3)
    plt.plot(t, pila)
    plt.title('Piłokształtny')
    plt.xlabel('Czas')
    plt.ylabel('Amplituda')
    plt.grid(True)

    plt.subplot(6, 1, 4)
    plt.plot(t, swiergot)
    plt.title('Świergotliwy')
    plt.xlabel('Czas')
    plt.ylabel('Amplituda')
    plt.grid(True)

    plt.subplot(6, 1, 5)
    plt.plot(t, superpozycja_sin_cos)
    plt.title('Superpozycja sinusa i cosinusa')
    plt.xlabel('Czas')
    plt.ylabel('Amplituda')
    plt.grid(True)

    plt.subplot(6, 1, 6)
    plt.plot(t, dirac)
    plt.title('Delta Diraca')
    plt.xlabel('Czas')
    plt.ylabel('Amplituda')
    plt.grid(True)

    plt.tight_layout()
    plt.show()




    # tworzenie subplotow spektogramow
    plt.figure(figsize=(10, 8))
    plt.subplot(6,1,1)
    f, t, Sxx = sig.spectrogram(sinus, fs=czest_prob, window=np.hamming(512), nperseg=512, noverlap=256,
                                scaling='spectrum', mode='magnitude')
    plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud')
    plt.colorbar(label='Amplituda [dB]')
    plt.title('Spektrogram sygnału sinusa')
    plt.xlabel('Czas [s]')
    plt.ylabel('Częstotliwość [Hz]')
    plt.ylim([0, 50])  # Ograniczenie zakresu częstotliwości dla lepszej widoczności

    plt.subplot(6,1,2)
    f, t, Sxx = sig.spectrogram(prosto, fs=czest_prob, window=np.hamming(512), nperseg=512, noverlap=256,
                                scaling='spectrum', mode='magnitude')
    plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud')
    plt.colorbar(label='Amplituda [dB]')
    plt.title('Spektrogram sygnału prostokatnego')
    plt.xlabel('Czas [s]')
    plt.ylabel('Częstotliwość [Hz]')
    plt.ylim([0, 60])  # Ograniczenie zakresu częstotliwości dla lepszej widoczności

    plt.subplot(6,1,3)
    f, t, Sxx = sig.spectrogram(pila, fs=czest_prob, window=np.hamming(512), nperseg=512, noverlap=256,
                                scaling='spectrum', mode='magnitude')
    plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud')
    plt.colorbar(label='Amplituda [dB]')
    plt.title('Spektrogram sygnału Pily')
    plt.xlabel('Czas [s]')
    plt.ylabel('Częstotliwość [Hz]')
    plt.ylim([0, 200])  # Ograniczenie zakresu częstotliwości dla lepszej widoczności

    plt.subplot(6, 1, 4)
    f, t, Sxx = sig.spectrogram(swiergot, fs=czest_prob, window=np.hamming(512), nperseg=512, noverlap=256,
                                scaling='spectrum', mode='magnitude')
    plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud')
    plt.colorbar(label='Amplituda [dB]')
    plt.title('Spektrogram sygnału chirp')
    plt.xlabel('Czas [s]')
    plt.ylabel('Częstotliwość [Hz]')
    plt.ylim([0, 200])  # Ograniczenie zakresu częstotliwości dla lepszej widoczności

    plt.subplot(6, 1, 5)
    f, t, Sxx = sig.spectrogram(superpozycja_sin_cos, fs=czest_prob, window=np.hamming(512), nperseg=512, noverlap=256,
                                scaling='spectrum', mode='magnitude')
    plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud')
    plt.colorbar(label='Amplituda [dB]')
    plt.title('Spektrogram sygnału superpozycji')
    plt.xlabel('Czas [s]')
    plt.ylabel('Częstotliwość [Hz]')
    plt.ylim([0, 50])  # Ograniczenie zakresu częstotliwości dla lepszej widoczności

    plt.subplot(6, 1, 6)
    czest_prob = 1000
    amplituda = 1

    t = np.linspace(-1, 1, czest_prob)
    dirac = np.zeros_like(t)
    dirac[(t >= -0.05) & (t <= 0.05)] = 1 * amplituda

    plt.specgram( dirac, Fs=czest_prob)
    plt.title('(dzialajacy) spektogram delty diraca')
    plt.xlabel('czsas')
    plt.ylabel('czestotliwosc (Hz)')
    plt.colorbar(label='amplituda (dB)')

    plt.tight_layout()
    plt.show()

    wav_fs, gitara = wavfile.read('4thg.wav')
    def load_audio(file_path, sample_rate=wav_fs):
        audio, sr = librosa.load(file_path, sr=sample_rate, mono=False)
        return audio, sr

    def calculate_spectrogram(audio, sample_rate):
        mono_audio = librosa.to_mono(audio)
        stft = librosa.stft(mono_audio)
        spectrogram = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
        return spectrogram

    def calculate_stereo_image(audio):
        left_channel = audio[0, :]
        right_channel = audio[1, :]
        stereo_image = left_channel - right_channel
        return stereo_image

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))

    def visualize_audio(audio, sample_rate):
        spectrogram = calculate_spectrogram(audio, sample_rate)
        stereo_image = calculate_stereo_image(audio)
        # Plot the spectrogram
        plt.figure(figsize=(12, 6))
        plt.subplot(2, 1, 1)
        librosa.display.specshow(spectrogram, sr=sample_rate, x_axis="time", y_axis="log")
        plt.colorbar(format="%+2.0f dB")
        plt.title("Spectrogram")
        # Plot the stereo image
        plt.subplot(2, 1, 2)
        librosa.display.waveshow(stereo_image, sr=sample_rate, x_axis="time")
        plt.title("Stereo Image")
        plt.tight_layout()
        plt.show()

    def animate(i):
        ax1.clear()
        ax2.clear()
        # Update spectrogram plot
        librosa.display.specshow(spectrogram[:, :max(1, i)], sr=sample_rate, x_axis="time", y_axis="log", ax=ax1)
        ax1.set_title("Spectrogram")
        # Update stereo image plot
        librosa.display.waveshow(stereo_image[:max(1, i)], sr=sample_rate, axis="time", ax=ax2)
        ax2.set_title("Stereo Image")



    # pelny obrazek
    if __name__ == "__main__":
        file_path = "C:/Users/Bartek/Documents/strdanych/strumieniefinal/4thg.wav"
        audio, sample_rate = load_audio(file_path)
        visualize_audio(audio, sample_rate)
    """
    #animacja
    if __name__ == "__main__":
        file_path = "C:/Users/Bartek/Documents/strdanych/strumieniefinal/4thg.wav"
        audio, sample_rate = load_audio(file_path)

        spectrogram = calculate_spectrogram(audio, sample_rate)
        stereo_image = calculate_stereo_image(audio)

        frames = spectrogram.shape[1]

        ani = FuncAnimation(fig, animate, frames=frames, interval=20, blit=False)
        plt.tight_layout()
        plt.show()
"""
def z1():
    root = tk.Tk()
    root.title("Manipulacja parametrami")

    label_amplituda = tk.Label(root, text='Amplituda:')
    label_amplituda.grid(row=0, column=0)
    slider_amplituda = tk.Scale(root, from_=1.0, to=20.0, orient=tk.HORIZONTAL, resolution=1.0)
    slider_amplituda.grid(row=0, column=1)

    label_frekwencja = tk.Label(root, text='Częstotliwość:')
    label_frekwencja.grid(row=1, column=0)
    slider_frekwencja = tk.Scale(root, from_=2.0, to=50.0, orient=tk.HORIZONTAL, resolution=1.0)
    slider_frekwencja.grid(row=1, column=1)

    label_czest_prob = tk.Label(root, text='Częstotliwość próbkowania:')
    label_czest_prob.grid(row=2, column=0)
    slider_czest_prob = tk.Scale(root, from_=1000, to=2000, orient=tk.HORIZONTAL, resolution=100)
    slider_czest_prob.grid(row=2, column=1)

    label_t_trwania = tk.Label(root, text='Czas trwania:')
    label_t_trwania.grid(row=3, column=0)
    slider_t_trwania = tk.Scale(root, from_=5.0, to=20.0, orient=tk.HORIZONTAL, resolution=1.0)
    slider_t_trwania.grid(row=3, column=1)

    button_generate = ttk.Button(root, text="Generuj sygnał", command=lambda: zad1(slider_amplituda.get(), slider_frekwencja.get(), slider_czest_prob.get(), slider_t_trwania.get()))
    button_generate.grid(row=4, columnspan=2)

    root.mainloop()


def zad2(amplituda, frekwencja, czest_prob, t_trwania):
    t = np.linspace(0, t_trwania, int(czest_prob * t_trwania), endpoint=False)
    swiergot = amplituda * sig.chirp(2 * np.pi * frekwencja * t, 6, 10, 1)

    # Dekompozycja sygnału przy użyciu EMD
    emd = EMD()
    IMFs = emd(swiergot)

    # Obliczenie widma każdej z modów
    spectra = np.abs(np.fft.fft(IMFs, axis=1))[:, :czest_prob // 2]
    frequencies = np.fft.fftfreq(czest_prob, 1 / czest_prob)[:czest_prob // 2]

    # Wyświetlenie widma każdej z modów
    plt.figure(figsize=(10, 6))


    plt.subplot(3,1,1)
    plt.plot(t, swiergot)
    plt.title('Świergotliwy')
    plt.xlabel('Czas')
    plt.ylabel('Amplituda')
    plt.grid(True)


    plt.subplot(3, 1, 2)
    for i, mode in enumerate(IMFs):
        plt.plot(t, mode, label=f"Mode {i + 1}")
    plt.title("Mode'y wyodrębniona przez EMD")
    plt.xlabel("Czas [s]")
    plt.ylabel("Amplituda")
    plt.legend()
    plt.grid(True)

    plt.subplot(3,1,3)
    for i, mode in enumerate(spectra):
        plt.plot(frequencies, mode, label=f"Mode {i + 1}")
    plt.title("Widmo każdej z modów")
    plt.xlabel("Częstotliwość [Hz]")
    plt.ylabel("Amplituda")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def z2():
    root = tk.Tk()
    root.title("Manipulacja parametrami")

    label_amplituda = tk.Label(root, text='Amplituda:')
    label_amplituda.grid(row=0, column=0)
    slider_amplituda = tk.Scale(root, from_=1.0, to=20.0, orient=tk.HORIZONTAL, resolution=1.0)
    slider_amplituda.grid(row=0, column=1)

    label_frekwencja = tk.Label(root, text='Częstotliwość:')
    label_frekwencja.grid(row=1, column=0)
    slider_frekwencja = tk.Scale(root, from_=2.0, to=50.0, orient=tk.HORIZONTAL, resolution=1.0)
    slider_frekwencja.grid(row=1, column=1)

    label_czest_prob = tk.Label(root, text='Częstotliwość próbkowania:')
    label_czest_prob.grid(row=2, column=0)
    slider_czest_prob = tk.Scale(root, from_=1000, to=2000, orient=tk.HORIZONTAL, resolution=100)
    slider_czest_prob.grid(row=2, column=1)

    label_t_trwania = tk.Label(root, text='Czas trwania:')
    label_t_trwania.grid(row=3, column=0)
    slider_t_trwania = tk.Scale(root, from_=1.0, to=20.0, orient=tk.HORIZONTAL, resolution=1.0)
    slider_t_trwania.grid(row=3, column=1)

    button_generate = ttk.Button(root, text="Generuj sygnał", command=lambda: zad2(slider_amplituda.get(), slider_frekwencja.get(), slider_czest_prob.get(), slider_t_trwania.get()))
    button_generate.grid(row=4, columnspan=2)

    root.mainloop()


def zad3(amplituda1, amplituda2, amplituda3,
         frekwencja1, frekwencja2, frekwencja3,
         czest_prob, t_trwania):
    t = np.linspace(0, t_trwania, int(czest_prob * t_trwania), endpoint=False)

    sygnal = (
            amplituda1 * sig.chirp(t, f0=frekwencja1, f1=frekwencja1, t1=1, method='linear', phi=-90) +
            amplituda2 * sig.chirp(t, f0=frekwencja2, f1=frekwencja2, t1=1, method='linear', phi=0) +
            amplituda3 * np.sin(2 * np.pi * frekwencja3 * t)
    )


    emd = EMD()
    IMFs = emd(sygnal)
    spectra = np.abs(np.fft.fft(IMFs, axis=1))[:, :czest_prob // 2]
    frequencies = np.fft.fftfreq(czest_prob, 1 / czest_prob)[:czest_prob // 2]

    # Wyświetlenie sygnału
    plt.subplot(3, 1, 1)
    plt.plot(t, sygnal)
    plt.title('Sygnał (sinus + cosinus + sinus)')
    plt.xlabel('Czas')
    plt.ylabel('Amplituda')
    plt.grid(True)

    # Wyświetlenie modów wyodrębnionych przez EMD
    plt.subplot(3, 1, 2)
    for i, mode in enumerate(IMFs):
        plt.plot(t, mode, label=f"Mode {i + 1}")
    plt.title("Mode'y wyodrębniona przez EMD")
    plt.xlabel("Czas [s]")
    plt.ylabel("Amplituda")
    plt.legend()
    plt.grid(True)

    # Wyświetlenie widma każdej z modów
    plt.subplot(3, 1, 3)
    for i, mode in enumerate(spectra):
        plt.plot(frequencies, mode, label=f"Mode {i + 1}")
    plt.title("Widmo każdej z modów")
    plt.xlabel("Częstotliwość [Hz]")
    plt.ylabel("Amplituda")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def z3():
    root = tk.Tk()
    root.title("Manipulacja parametrami")

    label_amplituda1 = tk.Label(root, text='Amplituda 1:')
    label_amplituda1.grid(row=0, column=0)
    slider_amplituda1 = tk.Scale(root, from_=1.0, to=20.0, orient=tk.HORIZONTAL, resolution=1.0)
    slider_amplituda1.grid(row=0, column=1)

    label_amplituda2 = tk.Label(root, text='Amplituda 2:')
    label_amplituda2.grid(row=1, column=0)
    slider_amplituda2 = tk.Scale(root, from_=1.0, to=20.0, orient=tk.HORIZONTAL, resolution=1.0)
    slider_amplituda2.grid(row=1, column=1)

    label_amplituda3 = tk.Label(root, text='Amplituda 3:')
    label_amplituda3.grid(row=2, column=0)
    slider_amplituda3 = tk.Scale(root, from_=1.0, to=20.0, orient=tk.HORIZONTAL, resolution=1.0)
    slider_amplituda3.grid(row=2, column=1)

    label_frekwencja1 = tk.Label(root, text='Częstotliwość 1:')
    label_frekwencja1.grid(row=3, column=0)
    slider_frekwencja1 = tk.Scale(root, from_=2.0, to=50.0, orient=tk.HORIZONTAL, resolution=1.0)
    slider_frekwencja1.grid(row=3, column=1)

    label_frekwencja2 = tk.Label(root, text='Częstotliwość 2:')
    label_frekwencja2.grid(row=4, column=0)
    slider_frekwencja2 = tk.Scale(root, from_=2.0, to=50.0, orient=tk.HORIZONTAL, resolution=1.0)
    slider_frekwencja2.grid(row=4, column=1)

    label_frekwencja3 = tk.Label(root, text='Częstotliwość 3:')
    label_frekwencja3.grid(row=5, column=0)
    slider_frekwencja3 = tk.Scale(root, from_=2.0, to=50.0, orient=tk.HORIZONTAL, resolution=1.0)
    slider_frekwencja3.grid(row=5, column=1)

    label_czest_prob = tk.Label(root, text='Częstotliwość próbkowania:')
    label_czest_prob.grid(row=6, column=0)
    slider_czest_prob = tk.Scale(root, from_=1000, to=2000, orient=tk.HORIZONTAL, resolution=100)
    slider_czest_prob.grid(row=6, column=1)

    label_t_trwania = tk.Label(root, text='Czas trwania:')
    label_t_trwania.grid(row=7, column=0)
    slider_t_trwania = tk.Scale(root, from_=5.0, to=20.0, orient=tk.HORIZONTAL, resolution=1.0)
    slider_t_trwania.grid(row=7, column=1)

    button_generate = ttk.Button(root, text="Generuj sygnał",
                                 command=lambda: zad3(slider_amplituda1.get(), slider_amplituda2.get(),
                                                      slider_amplituda3.get(),
                                                      slider_frekwencja1.get(), slider_frekwencja2.get(),
                                                      slider_frekwencja3.get(),
                                                      slider_czest_prob.get(), slider_t_trwania.get()))
    button_generate.grid(row=8, columnspan=2)

    root.mainloop()


def zad4(file_path):
    data = pd.read_csv(file_path, sep=',')

    signal = data['Amplituda'].values
    time = data['time'].values

    time_diff = np.diff(time)
    sampling_frequency = 1 / time_diff.mean()

    emd = EMD()
    imfs = emd(signal, max_imf=5)

    plt.figure(figsize=(10, 6))

    plt.subplot(3, 1, 1)
    plt.plot(time, signal)
    plt.title('Sygnal')
    plt.xlabel('Czas')
    plt.ylabel('Amplituda')
    plt.grid(True)

    # IMFs
    plt.subplot(3, 1, 2)
    for i, mode in enumerate(imfs):
        plt.plot(time, mode, label=f"Mode {i + 1}")
    plt.title("Mode'y")
    plt.xlabel("Czas")
    plt.ylabel("Amplituda")
    plt.legend()
    plt.grid(True)

    # Spectra
    plt.subplot(3, 1, 3)
    frequencies = np.fft.fftfreq(signal.size, 1 / sampling_frequency)[:signal.size // 2]
    spectra = np.abs(np.fft.fft(imfs, axis=1))
    for i, mode in enumerate(spectra):
        plt.plot(frequencies, mode[:len(frequencies)], label=f"Mode {i + 1}")
    plt.title("Widmo mode'ów")
    plt.xlabel("Częstotliwość")
    plt.ylabel("Amplituda")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def z4():
    root = tk.Tk()
    root.title("Manipulacja parametrami")

    version_label = ttk.Label(root, text="podaj sciezke do pliku")
    version_label.pack()
    version_entry = ttk.Entry(root)
    version_entry.pack()
    version_entry.insert(0, '1')

    button_generate3 = ttk.Button(root, text="zad4",
                                  command=lambda: zad4(str(version_entry.get())))
    button_generate3.pack()

    root.mainloop()
#z1()
#z2()
#z3()
z4()