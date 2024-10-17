import scipy.signal as sig
import matplotlib.pyplot as plt
import numpy as np
import time
import math
import pandas as pd
import tkinter as tk
from tkinter import ttk
import ipywidgets as widgets
from IPython.display import display

#zadanie 1

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
    slider_czest_prob = tk.Scale(root, from_=100, to=2000, orient=tk.HORIZONTAL, resolution=100)
    slider_czest_prob.grid(row=2, column=1)

    label_t_trwania = tk.Label(root, text='Czas trwania:')
    label_t_trwania.grid(row=3, column=0)
    slider_t_trwania = tk.Scale(root, from_=0.5, to=2.0, orient=tk.HORIZONTAL, resolution=0.1)
    slider_t_trwania.grid(row=3, column=1)

    button_generate = ttk.Button(root, text="Generuj sygnał", command=lambda: zad1(slider_amplituda.get(), slider_frekwencja.get(), slider_czest_prob.get(), slider_t_trwania.get()))
    button_generate.grid(row=4, columnspan=2)

    root.mainloop()

#zadanie 2
def z2():
    dane = pd.read_csv('sygnal1.csv', sep=';')
    #print(dane.head())

    plt.plot(dane['time'], dane['signal1'])
    plt.xlabel('Czas')
    plt.ylabel('Amplituda')
    plt.title('Pobrany sygnal z pliku CSV')
    plt.grid(True)
    plt.show()

#zadanie 3
def zad3(amplituda,frekwencja,czest_prob,t_trwania):

    t = np.linspace(0, t_trwania, int(czest_prob * t_trwania), endpoint=False)

    # tworzenie sygnałów
    swiergot = amplituda * sig.chirp(2*np.pi * frekwencja * t,6,1,1)

    plt.figure(figsize=(10,6))
    plt.subplot(1,1,1)
    plt.plot(t, swiergot)
    plt.title('Świergotliwy')
    plt.xlabel('Czas')
    plt.ylabel('Amplituda')
    plt.grid(True)
    plt.show()

    #zapisywanie
    dane = pd.DataFrame({
        'time': t,
        'Amplituda': swiergot
    })
    dane.to_csv('syngal2.csv')

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
    slider_czest_prob = tk.Scale(root, from_=100, to=2000, orient=tk.HORIZONTAL, resolution=100)
    slider_czest_prob.grid(row=2, column=1)

    label_t_trwania = tk.Label(root, text='Czas trwania:')
    label_t_trwania.grid(row=3, column=0)
    slider_t_trwania = tk.Scale(root, from_=0.5, to=2.0, orient=tk.HORIZONTAL, resolution=0.1)
    slider_t_trwania.grid(row=3, column=1)

    button_generate = ttk.Button(root, text="Generuj sygnał", command=lambda: zad3(slider_amplituda.get(), slider_frekwencja.get(), slider_czest_prob.get(), slider_t_trwania.get()))
    button_generate.grid(row=4, columnspan=2)

    root.mainloop()

#zadanie 4
def zad4(ile_probek,ile_koszykow):

    syg_rand = np.random.rand(ile_probek)
    syg_randn = np.random.randn(ile_probek)

    plt.figure(figsize=(10,6))
    plt.subplot(2,1,1)
    plt.hist(syg_rand,bins=ile_koszykow, color='red', histtype='bar', edgecolor = 'k')
    plt.title('Histogram dla rand')
    plt.xlabel('Wartosci')
    plt.ylabel('Częstość')

    plt.subplot(2,1,2)
    plt.hist(syg_randn, bins=ile_koszykow, color = 'blue', histtype='bar', edgecolor = 'k')
    plt.title('Histogram dla randn')
    plt.xlabel('Wartosci')
    plt.ylabel('Częstość')

    plt.show()

def z4():
    root = tk.Tk()
    root.title("Manipulacja parametrami")

    label_probki = tk.Label(root, text='Liczba próbek:')
    label_probki.grid(row=0, column=0)
    slider_probki = tk.Scale(root, from_=10, to=5000, orient=tk.HORIZONTAL, resolution=10)
    slider_probki.grid(row=0, column=1)

    label_koszyki = tk.Label(root, text='Ilość koszyków:')
    label_koszyki.grid(row=2, column=0)
    slider_koszyki = tk.Scale(root, from_=1, to=50, orient=tk.HORIZONTAL, resolution=1)
    slider_koszyki.grid(row=2, column=1)

    button_generate = ttk.Button(root, text="Generuj wykresy", command=lambda: zad4(slider_probki.get(), slider_koszyki.get()))
    button_generate.grid(row=3, columnspan=2)

    root.mainloop()
#zadanie 5
def zad5(ile_koszykow, ile_probek, s1, o1, s2, o2, s3, o3):
    ile_koszykow = int(ile_koszykow)
    ile_probek = int(ile_probek)

    normalny1 = np.random.normal(s1, o1, size=ile_probek)
    normalny2 = np.random.normal(s2, o2, size=ile_probek)
    normalny3 = np.random.normal(s3, o3, size=ile_probek)

    plt.figure(figsize=(10, 6))

    plt.subplot(3, 1, 1)
    plt.hist(normalny1, bins=ile_koszykow, color='red', histtype='bar', edgecolor='k')
    plt.title(f'Histogram dla sredniej={s1} i odchyleniu={o1}')
    plt.xlabel('Wartosci')
    plt.ylabel('Częstość')

    plt.subplot(3, 1, 2)
    plt.hist(normalny2, bins=ile_koszykow, color='blue', histtype='bar', edgecolor='k')
    plt.title(f'Histogram dla sredniej={s2} i odchyleniu={o2}')
    plt.xlabel('Wartosci')
    plt.ylabel('Częstość')

    plt.subplot(3, 1, 3)
    plt.hist(normalny3, bins=ile_koszykow, color='green', histtype='bar', edgecolor='k')
    plt.title(f'Histogram dla sredniej={s3} i odchyleniu={o3}')
    plt.xlabel('Wartosci')
    plt.ylabel('Częstość')

    plt.tight_layout()
    plt.show()

def z5():
    #suwaki w tkinterze
    root = tk.Tk()
    root.title("Manipulacja parametrami")

    label_koszyki = tk.Label(root, text='Ilość koszyków:')
    label_koszyki.grid(row=0, column=0)
    slider_koszyki = tk.Scale(root, from_=1, to=50, orient=tk.HORIZONTAL, resolution=1)
    slider_koszyki.grid(row=0, column=1)

    label_probki = tk.Label(root, text='Ilość próbek:')
    label_probki.grid(row=1, column=0)
    slider_probki = tk.Scale(root, from_=100, to=10000, orient=tk.HORIZONTAL, resolution=100)
    slider_probki.grid(row=1, column=1)

    label_s1 = tk.Label(root, text='Średnia 1:')
    label_s1.grid(row=2, column=0)
    slider_s1 = tk.Scale(root, from_=-5, to=5, orient=tk.HORIZONTAL, resolution=0.1)
    slider_s1.grid(row=2, column=1)

    label_o1 = tk.Label(root, text='Odchylenie 1:')
    label_o1.grid(row=3, column=0)
    slider_o1 = tk.Scale(root, from_=0.1, to=5, orient=tk.HORIZONTAL, resolution=0.1)
    slider_o1.grid(row=3, column=1)

    label_s2 = tk.Label(root, text='Średnia 2:')
    label_s2.grid(row=4, column=0)
    slider_s2 = tk.Scale(root, from_=-5, to=5, orient=tk.HORIZONTAL, resolution=0.1)
    slider_s2.grid(row=4, column=1)

    label_o2 = tk.Label(root, text='Odchylenie 2:')
    label_o2.grid(row=5, column=0)
    slider_o2 = tk.Scale(root, from_=0.1, to=5, orient=tk.HORIZONTAL, resolution=0.1)
    slider_o2.grid(row=5, column=1)

    label_s3 = tk.Label(root, text='Średnia 3:')
    label_s3.grid(row=6, column=0)
    slider_s3 = tk.Scale(root, from_=-5, to=5, orient=tk.HORIZONTAL, resolution=0.1)
    slider_s3.grid(row=6, column=1)

    label_o3 = tk.Label(root, text='Odchylenie 3:')
    label_o3.grid(row=7, column=0)
    slider_o3 = tk.Scale(root, from_=0.1, to=5, orient=tk.HORIZONTAL, resolution=0.1)
    slider_o3.grid(row=7, column=1)

    button_generate = ttk.Button(root, text="Generuj histogram", command=lambda: zad5(slider_koszyki.get(), slider_probki.get(), slider_s1.get(), slider_o1.get(), slider_s2.get(), slider_o2.get(), slider_s3.get(), slider_o3.get()))
    button_generate.grid(row=8, columnspan=2)

    root.mainloop()

#zadanie 6
def zad6(liczba_probek, czestotliwosc_probkowania, ile_koszykow):
    bialy_szum = np.random.normal(0, 1, size=liczba_probek)
    cz_sz = np.cumsum(bialy_szum)
    t = np.arange(0, liczba_probek) / czestotliwosc_probkowania

    plt.figure(figsize=(10, 8))

    #szum bialy
    plt.subplot(3, 1, 1)
    plt.plot(t, bialy_szum, color='blue')
    plt.title('Szum biały (rozklad normalny o mi=0, std=1)')
    plt.xlabel('Czas')
    plt.ylabel('Amplituda')
    plt.grid(True)
    #szum czerwony
    plt.subplot(3, 1, 2)
    plt.plot(t, cz_sz, color='red')
    plt.title('Szum czerwony (Browna)')
    plt.xlabel('Czas')
    plt.ylabel('Amplituda')
    plt.grid(True)
    #histogram szumu cz.
    plt.subplot(3, 1, 3)
    plt.hist(cz_sz, bins=ile_koszykow, color='red', edgecolor='black')
    plt.title('Histogram dla szumu czerwonego (Browna)')
    plt.xlabel('Wartość')
    plt.ylabel('Częstość')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def z6():
    root = tk.Tk()
    root.title("Manipulacja parametrami")

    label_probki = tk.Label(root, text='Liczba próbek:')
    label_probki.grid(row=0, column=0)
    slider_probki = tk.Scale(root, from_=10, to=1000, orient=tk.HORIZONTAL, resolution=10)
    slider_probki.grid(row=0, column=1)

    label_probkowanie = tk.Label(root, text='Częstotliwość próbkowania:')
    label_probkowanie.grid(row=1, column=0)
    slider_probkowanie = tk.Scale(root, from_=10, to=1000, orient=tk.HORIZONTAL, resolution=10)
    slider_probkowanie.grid(row=1, column=1)

    label_koszyki = tk.Label(root, text='Ilość koszyków:')
    label_koszyki.grid(row=2, column=0)
    slider_koszyki = tk.Scale(root, from_=1, to=50, orient=tk.HORIZONTAL, resolution=1)
    slider_koszyki.grid(row=2, column=1)

    button_generate = ttk.Button(root, text="Generuj wykresy", command=lambda: zad6(slider_probki.get(), slider_probkowanie.get(), slider_koszyki.get()))
    button_generate.grid(row=3, columnspan=2)

    root.mainloop()

#zadanie7
def zad7(rozmiar):

    bialy_szum = np.random.normal(0, 1, (rozmiar, rozmiar))

    szum_czerwony = np.cumsum(np.cumsum(bialy_szum, axis=0), axis=1)

    #czasem potrzebna normalizacja bo nic nie widac
    '''
    szum_czerwony -= np.mean(szum_czerwony)
    szum_czerwony /= np.max(np.abs(szum_czerwony))
    
    '''
    plt.figure(figsize=(10,6))
    plt.subplot(1,1,1)
    plt.imshow(szum_czerwony, cmap='hot', interpolation='nearest')
    plt.title('Dwuwymiarowy szum czerwony (Browna)')
    plt.colorbar(label='Amplituda')
    plt.show()

def z7():
    root = tk.Tk()
    root.title("Manipulacja parametrami")

    label_probki = tk.Label(root, text='rozmiar:')
    label_probki.grid(row=0, column=0)
    slider_probki = tk.Scale(root, from_=10, to=500, orient=tk.HORIZONTAL, resolution=10)
    slider_probki.grid(row=0, column=1)

    button_generate = ttk.Button(root, text="Generuj wykresy", command=lambda: zad7(slider_probki.get()))
    button_generate.grid(row=3, columnspan=2)

    root.mainloop()



z1()
#z2()
#z3()
#z4()
#z5()
#z6()
#z7()