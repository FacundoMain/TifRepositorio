# -*- coding: utf-8 -*-
"""
Created on Thu May 23 00:08:09 2024

@author: nahue
"""

import os
import numpy as np
from scipy.signal import firwin, lfilter, butter, filtfilt
from scipy.fftpack import fft
import matplotlib.pyplot as plt

def plot_time_signal(data, fs=10, title="Señal en el Tiempo"):
    N = len(data)  # Número total de muestras
    T = 1/fs  # Periodo de muestreo
    t = np.linspace(0, N*T, N, endpoint=False)  # Vector de tiempo
    plt.figure(figsize=(10, 6))
    plt.plot(t, data, label='Señal x[n]')
    plt.title(title)
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Amplitud')    
    plt.grid(True)
    plt.legend()
    plt.show()



def fir_lowpass_filter(data, cutoff=0.1, fs=10, numtaps=31):
    nyq = 0.5 * fs  # Frecuencia de Nyquist
    normal_cutoff = cutoff / nyq
    # Obtiene los coeficientes del filtro FIR
    b = firwin(numtaps, normal_cutoff)
    # Aplica el filtro
    y = lfilter(b, 1.0, data)
    return y

def butter_lowpass_filter(data, cutoff= 0.1, fs=10, order=10):
    nyq = 0.5 * fs  # Frecuencia de Nyquist
    normal_cutoff = cutoff / nyq
    # Obtiene los coeficientes del filtro
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    # Aplica el filtro
    y = filtfilt(b, a, data)
    return y

def load_samples(filename):
    data = np.loadtxt(filename)
    return data

def derivative(data, fs=10):
    dt = 1/fs
    # Diferencia centrada excepto para los bordes
    d_data = np.diff(data) / dt
    return d_data

def remove_dc(data):
    return data - np.mean(data)

def plot_frequency_spectrum(data, fs=10, title="Espectro de Frecuencia"):
    # Número de puntos
    N = len(data)
    # FFT
    yf = fft(data)
    xf = np.linspace(0.0, fs/2, N//2)
    plt.figure(figsize=(10, 6))
    plt.plot(xf, 2.0/N * np.abs(yf[:N//2]))
    plt.title(title)
    plt.xlabel('Frecuencia (Hz)')
    plt.ylabel('Amplitud')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    plt.close('all')
    folder = "Registros"  # Carpeta que contiene los archivos

    for filename in os.listdir(folder):
        if filename.endswith(".txt"):  # Asegurarse de que solo se procesen archivos .txt
            filepath = os.path.join(folder, filename)
            
            # Carga de muestras
            samples = load_samples(filepath)
            
            # Graficar el espectro de frecuencia original
            #plot_frequency_spectrum(samples, fs=10, title=f"Espectro de Frecuencia Original - {filename}")
            
            # Remover la componente de 0 Hz
            samples_no_dc = remove_dc(samples)
            
            # Aplicar filtro pasa bajo
            filtered_samples = fir_lowpass_filter(samples_no_dc)
            #filtered_samples = butter_lowpass_filter(samples_no_dc)
            
            # # Graficar el espectro de frecuencia filtrado
            # plot_frequency_spectrum(filtered_samples, fs=10, title=f"Espectro de Frecuencia Filtrado - {filename}")
            
            # # Calcular la derivada
            # derivative_samples = derivative(filtered_samples)
            
            # # Interpolación lineal (se hace implícitamente al graficar)
            # plt.figure(figsize=(10, 6))
            # plt.plot(np.linspace(0, len(derivative_samples)/10, len(derivative_samples)), derivative_samples, marker='o', linestyle='-', label='Derivada (interpolada para graficar)')
            # plt.title(f"Derivada de las muestras filtradas - {filename}")
            # plt.xlabel("Tiempo (s)")
            # plt.ylabel("Amplitud")
            # plt.legend()
            # plt.grid(True)
            # plt.show()
            
            plot_time_signal(filtered_samples, fs=10, title=f"Señal Original en el Tiempo - {filename}")
