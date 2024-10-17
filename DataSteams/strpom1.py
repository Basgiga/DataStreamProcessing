import numpy as np
import pandas as pd
import scipy.signal as sig

# Function to generate a chirp signal
def generate_chirp_signal(amplitude, frequency, sampling_rate, duration):
    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
    signal = amplitude * sig.chirp(2 * np.pi * frequency * t, 6, duration, 1)
    return t, signal

# Function to add noise to a signal
def add_noise(signal, noise_std):
    noise = np.random.normal(0, noise_std, len(signal))
    noisy_signal = signal + noise
    return noisy_signal

# Parameters
amplitude = 1.0
frequency = 10.0
sampling_rate = 1000  # Hz
duration = 1.0  # seconds
noise_std = 0.1

# Generate chirp signal
t, signal = generate_chirp_signal(amplitude, frequency, sampling_rate, duration)

# Save signal to CSV file
df_signal = pd.DataFrame({'Time': t, 'Signal': signal})
df_signal.to_csv('signal4.csv', index=False)

# Add noise to signal
noisy_signal = add_noise(signal, noise_std)

# Save noisy signal to CSV file
df_noisy_signal = pd.DataFrame({'Time': t, 'NoisySignal': noisy_signal})
df_noisy_signal.to_csv('ns.csv', index=False)