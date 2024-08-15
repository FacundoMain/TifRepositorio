import os
import numpy as np
from scipy.signal import firwin, lfilter, butter, filtfilt, savgol_filter, find_peaks 
from scipy import fft,stats
import matplotlib.pyplot as plt
import math
import scipy.stats as stats
from scipy.interpolate import CubicSpline
from scipy.optimize import curve_fit


def fir_lowpass_filter(data, cutoff=0.1, fs=10, numtaps=31):
    '''
    Aplica un filtro pasabajos FIR a una señal en el dominio del tiempo.

    Parámetros
    ----------
    data : array-like
        Vector que almacena la magnitud de la señal a filtrar.
    cutoff : float, opcional
        Frecuencia de corte del filtro (normalizada respecto a la frecuencia de Nyquist). El valor predeterminado es 0.1.
    fs : float, opcional
        Frecuencia de muestreo. El valor predeterminado es 10.
    numtaps : int, opcional
        Número de coeficientes del filtro FIR. El valor predeterminado es 31.

    Returns
    -------
    senial: array-like
        Vector con la señal filtrada.

   '''
   
    nyq = 0.5 * fs  # Frecuencia de Nyquist
    normal_cutoff = cutoff / nyq
    # Obtiene los coeficientes del filtro FIR
    b = firwin(numtaps, normal_cutoff)
    # Aplica el filtro
    senial = lfilter(b, 1.0, data)
    
    for i in range(30):
        senial[i] = senial[40]
    
    return senial

def load_samples(filename):
    '''
    Carga datos desde un archivo de texto.

    Parámetros
    ----------
    filename : str
        Ruta al archivo de datos.

    Returns
    -------
    data : array-like
        Vector con los datos cargados.
    '''
    data = np.loadtxt(filename)
    data = 1/data *1e6
   
    return data

def moving_average(data, window_size= 60):
    '''
    Aplica un filtro de media móvil a una señal en el dominio del tiempo.

    Parámetros
    ----------
    data : array-like
        Vector que almacena la magnitud de la señal.
    window_size : int, opcional
        Ancho de la ventana del filtro. El valor predeterminado es 400.

    Returns
    -------
    senial: array-like
        Vector con la señal filtrada.
    '''
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')


def remove_dc(data):
    '''
    Elimina la componente de corriente continua (DC) de una señal en el dominio del tiempo.

    Parámetros
    ----------
    data : array-like
        Vector que almacena la magnitud de la señal.

    Returns
    -------
    smoothed_data : array-like
        Vector con la señal filtrada (sin componente DC).
    '''
    fft_data = np.fft.fft(data)
    # Considera otras estrategias para eliminar la componente DC
    # (por ejemplo, restando la media antes de aplicar la transformada de Fourier)
    filtered_data = np.fft.ifft(fft_data).real
     
    smoothed_data = moving_average(filtered_data)
    
    if len(smoothed_data) == len(data):
        smoothed_data = smoothed_data[:-1]
    
    return smoothed_data


def normalizador(data):
    '''
    Normaliza una señal en el dominio del tiempo al rango [0, 1].

    Parámetros
    ----------
    data : array-like
        Vector que almacena la magnitud de la señal.

    Returns
    -------
    dataNorm : array-like
        Vector con la señal normalizada.
    '''
    delta = max(data) - min(data)
    dataNorm = (data - min(data)) / delta

    return dataNorm

def promedioPorGenero(all_filtered_signals):
    '''
    Calcula los promedios de dos grupos de señales: género femenino y género masculino.

    Parámetros
    ----------
    all_filtered_signals : list of tuples
        Lista de tuplas donde cada tupla contiene (nombre, señal normalizada).

    Returns
    -------
    promFemenino, promMasculino : tuple
        Tupla con dos vectores: promedio del grupo femenino y promedio del grupo masculino.
    '''
    N = len(all_filtered_signals[0][1])  # Número de señales

    # Inicializa variables
    cMasculinos = 0
    cFemeninos = 0
    
    promMasculino = np.zeros(N)
    promFemenino = np.zeros(N)
        
    # Calcula promedios y sumas de cuadrados
    for tupla in all_filtered_signals:
        partes = tupla[0].split('_')
        if '0' in partes:  # Género femenino
            for i in range(N):
                promFemenino[i] += tupla[1][i]
            cFemeninos += 1
        else:  # Género masculino
            for i in range(N):
                promMasculino[i] += tupla[1][i]
            cMasculinos += 1
    
    # Calcula promedios finales
    for i in range(N):
        promFemenino[i] /= cFemeninos
        promMasculino[i] /= cMasculinos
        
    return promFemenino, promMasculino


def desvioPorGenero(all_filtered_signals, promFemenino, promMasculino):
    '''
    Calcula los desvíos estándar de dos grupos de señales: género femenino y género masculino.

    Parámetros
    ----------
    all_filtered_signals : list of tuples
        Lista de tuplas donde cada tupla contiene (nombre, señal normalizada).
    promFemenino : array-like
        Vector con el promedio del grupo femenino.
    promMasculino : array-like
        Vector con el promedio del grupo masculino.

    Returns
    -------
    desvio_estandar_femenino, desvio_estandar_masculino : tuple
        Tupla con dos vectores: desvío estándar del grupo femenino y desvío estándar del grupo masculino.
    '''
    cMasculinos = 0
    cFemeninos = 0
    
    suma_cuadrados_masculino = np.zeros(N)
    suma_cuadrados_femenino = np.zeros(N)
    
    for tupla in all_filtered_signals:
        partes = tupla[0].split('_')
        if '0' in partes:  # Género femenino
            for i in range(N):
                suma_cuadrados_femenino[i] += (tupla[1][i] - promFemenino[i]) ** 2
            cFemeninos += 1
        else:  # Género masculino
            for i in range(N):
                suma_cuadrados_masculino[i] += (tupla[1][i] - promMasculino[i]) ** 2
            cMasculinos += 1
  
    desvio_estandar_femenino = np.sqrt(suma_cuadrados_femenino / cFemeninos)
    desvio_estandar_masculino = np.sqrt(suma_cuadrados_masculino / cMasculinos) 
    
    return desvio_estandar_femenino, desvio_estandar_masculino
   

#%%
if __name__ == "__main__":
    plt.close('all')
    folder = "Registros"  # Carpeta que contiene los archivos
  
    all_filtered_signals = []  # Para almacenar todas las señales filtradas
    seniales_sin_norm = []
    fs = 10  # Frecuencia de muestreo

    for filename in os.listdir(folder):
        if filename.endswith(".txt"):  # Asegurarse de que solo se procesen archivos .txt
            filepath = os.path.join(folder, filename)
            
            # Carga de muestras
            samples = load_samples(filepath)
            
            # Remover la componente de 0 Hz
            samples_no_dc = remove_dc(samples)
            
            # Aplicar filtro pasa bajo
            filtered_samples = fir_lowpass_filter(samples_no_dc, fs=fs)
             
            # Guardo señales normalizada                      
            Senial = normalizador(filtered_samples)
            
            # Almacenar la señal filtrada
            all_filtered_signals.append((filename, Senial))
              
    #%% Promedio y desvio en hombres y mujeres total
   
    T = 1/fs
    N = len(all_filtered_signals[1][1])
    t = np.linspace(0, N*T, N, endpoint=False)
    
    promedioFemenino, promedioMasculino = promedioPorGenero(all_filtered_signals)
    desvio_femenino, desvio_masculino = desvioPorGenero(all_filtered_signals, promedioFemenino, promedioMasculino)
    
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle("Promedio y desvio estandar en todas las mediciones", fontsize = 18)
    ax1.set_title("Promedio  masculino", fontsize = 15)
    ax1.plot(t,promedioMasculino,'b', label = 'Promedio')
    ax1.plot(t,promedioMasculino + desvio_masculino, '--g', label = 'Desvio')
    ax1.plot(t,promedioMasculino - desvio_masculino, '--g')
    ax1.set_xlabel('Tiempo [s]', fontsize = 12)
    ax1.set_ylabel('Amplitud', fontsize = 12)
    ax1.set_xlim([0,len(t)/10])
    ax1.legend()
    ax1.grid(True)

    ax2.set_title("Promedio  femenino", fontsize = 15)
    ax2.plot(t,promedioFemenino,'r', label = 'Promedio')
    ax2.plot(t,(promedioFemenino + desvio_femenino), '--g', label = 'Desvio')
    ax2.plot(t,(promedioFemenino - desvio_femenino), '--g')
    ax2.set_xlabel('Tiempo [s]', fontsize = 12)
    ax2.set_ylabel('Amplitud', fontsize = 12)
    ax2.set_xlim([0,len(t)/10])
    ax2.legend()
    ax2.grid(True)
    
    
    
    #%% separo en secciones
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle("Promedio y desvio estandar en primer segmento", fontsize = 18)
    ax1.set_title("Promedio  masculino", fontsize = 15)
    ax1.plot(t,promedioMasculino,'b', label = 'Promedio')
    ax1.plot(t,promedioMasculino + desvio_masculino, '--g', label = 'Desvio')
    ax1.plot(t,promedioMasculino - desvio_masculino, '--g')
    ax1.set_xlabel('Tiempo [s]', fontsize = 12)
    ax1.set_ylabel('Amplitud', fontsize = 12)
    ax1.set_xlim([0,400])
    ax1.plot(115, promedioMasculino[1140], 'ro', markersize=4, label='Picos')
    ax1.legend()
    ax1.grid(True)

    ax2.set_title("Promedio  femenino", fontsize = 15)
    ax2.plot(t,promedioFemenino,'r', label = 'Promedio')
    ax2.plot(t,(promedioFemenino + desvio_femenino), '--g', label = 'Desvio')
    ax2.plot(t,(promedioFemenino - desvio_femenino), '--g')
    ax2.set_xlabel('Tiempo [s]', fontsize = 12)
    ax2.set_ylabel('Amplitud [S]', fontsize = 12)
    ax2.set_xlim([0,400])
    ax2.plot(115, promedioFemenino[1140], 'bo', markersize=4, label='Picos')
    ax2.legend()
    ax2.grid(True)
   
    
   #%% segmento 2
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle("Promedio y desvio estandar en el segundo segmento", fontsize = 18)
    ax1.set_title("Promedio  masculino", fontsize = 15)
    ax1.plot(t,promedioMasculino,'b', label = 'Promedio')
    ax1.plot(t,promedioMasculino + desvio_masculino, '--g', label = 'Desvio')
    ax1.plot(t,promedioMasculino - desvio_masculino, '--g')
    ax1.set_xlabel('Tiempo [s]', fontsize = 12)
    ax1.set_ylabel('Amplitud', fontsize = 12)
    ax1.set_xlim([401,1226])
    ax1.legend()
    ax1.grid(True)

    ax2.set_title("Promedio  femenino", fontsize = 15)
    ax2.plot(t,promedioFemenino,'r', label = 'Promedio')
    ax2.plot(t,(promedioFemenino + desvio_femenino), '--g', label = 'Desvio')
    ax2.plot(t,(promedioFemenino - desvio_femenino), '--g')
    ax2.set_xlabel('Tiempo [s]', fontsize = 12)
    ax2.set_ylabel('Amplitud', fontsize = 12)
    ax2.set_xlim([401,1226])
    ax2.legend()
    ax2.grid(True)
    
    #%% segmento 3
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle("Promedio y desvio estandar en el tercer segmento", fontsize = 18)
    ax1.set_title("Promedio  masculino", fontsize = 15)
    ax1.plot(t,promedioMasculino,'b', label = 'Promedio')
    ax1.plot(t,promedioMasculino + desvio_masculino, '--g', label = 'Desvio')
    ax1.plot(t,promedioMasculino - desvio_masculino, '--g')
    ax1.set_xlabel('Tiempo [s]', fontsize = 12)
    ax1.set_ylabel('Amplitud', fontsize = 12)
    ax1.set_xlim([1226,len(t)/10])
    ax1.legend()
    ax1.grid(True)

    ax2.set_title("Promedio  femenino", fontsize = 15)
    ax2.plot(t,promedioFemenino,'r', label = 'Promedio')
    ax2.plot(t,(promedioFemenino + desvio_femenino), '--g', label = 'Desvio')
    ax2.plot(t,(promedioFemenino - desvio_femenino), '--g')
    ax2.set_xlabel('Tiempo [s]', fontsize = 12)
    ax2.set_ylabel('Amplitud', fontsize = 12)
    ax2.set_xlim([1226,len(t)/10])
    ax2.legend()
    ax2.grid(True)
        
#%% 
    fig, (ax3, ax4) = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle("Promedio y desvio estandar en segundo segmento", fontsize = 18)
    ax3.set_title("Promedio  masculino", fontsize = 15)
    ax3.plot(t,promedioMasculino,'b', label = 'Promedio')
    ax3.plot(t,promedioMasculino + desvio_masculino, '--g', label = 'Desvio')
    ax3.plot(t,promedioMasculino - desvio_masculino, '--g')
    ax3.plot(115, promedioMasculino[1149], 'ro', markersize=4, label='Picos')
    ax3.plot(518, promedioMasculino[5150], 'ro', markersize=4)
    ax3.plot(630, promedioMasculino[6300], 'ro', markersize=4)
    ax3.plot(800, promedioMasculino[8000], 'ro', markersize=4)
    ax3.plot(1095, promedioMasculino[10950], 'ro', markersize=4)
    ax3.set_xlabel('Tiempo [s]', fontsize = 12)
    ax3.set_ylabel('Amplitud', fontsize = 12)
    ax3.set_xlim([401,1226])
    # ax3.set_xlim([0,400])
    ax3.legend()
    ax3.grid(True)

    ax4.set_title("Promedio  femenino", fontsize = 15)
    ax4.plot(t,promedioFemenino,'r', label = 'Promedio')
    ax4.plot(t,(promedioFemenino + desvio_femenino), '--g', label = 'Desvio')
    ax4.plot(t,(promedioFemenino - desvio_femenino), '--g')
    ax4.plot(115, promedioFemenino[1149], 'bo', markersize=4, label='Picos')
    ax4.plot(955, promedioFemenino[9580], 'bo', markersize=4)
    ax4.plot(790, promedioFemenino[7900], 'bo', markersize=4)
    ax4.plot(1105, promedioFemenino[11050], 'bo', markersize=4)
    ax4.plot(1015, promedioFemenino[10150], 'bo', markersize=4)
    ax4.plot(510, promedioFemenino[5100], 'bo', markersize=4)
    ax4.set_xlabel('Tiempo [s]', fontsize = 12)
    ax4.set_ylabel('Amplitud', fontsize = 12)
    ax4.set_xlim([401,1226])
    # ax4.set_xlim([0,400])
    ax4.legend()
    ax4.grid(True)
    

#%%
signal = promedioFemenino[4001:12250]
t_tend = t[4001:12250]


# Calcular la línea de tendencia
slope, intercept, r_value, p_value, std_err = stats.linregress(t_tend,signal)


# Ajustar un polinomio de tercer grado (cúbico)
trend = savgol_filter(signal, window_length=800, polyorder=3)

# Restar la línea de tendencia
detrended_signal = signal- trend


# Graficar los resultados
plt.figure(figsize=(10, 6))
plt.plot(t_tend, signal, label='Señal Original')
plt.plot(t_tend, trend, label='Línea de Tendencia', linestyle='--')
plt.plot(955, promedioFemenino[9580], 'bo', markersize=4)
plt.plot(790, promedioFemenino[7900], 'bo', markersize=4)
plt.plot(1105, promedioFemenino[11050], 'bo', markersize=4)
plt.plot(1015, promedioFemenino[10150], 'bo', markersize=4)
plt.plot(510, promedioFemenino[5100], 'bo', markersize=4)

plt.plot(t_tend, detrended_signal, label='Señal Sin Tendencia')
plt.legend()
plt.xlabel('Tiempo')
plt.ylabel('Amplitud')
plt.title('Restar Línea de Tendencia de una Señal')
plt.show()

#%%
signalM = promedioMasculino[4001:12250]
trendM = savgol_filter(signalM, window_length=1500, polyorder=5)
slope, intercept, r_value, p_value, std_err = stats.linregress(t_tend,signalM)
detrended_signal_M = signalM- trendM

plt.figure(figsize=(10, 6))
plt.plot(t_tend, signalM, label='Señal Original')
plt.plot(t_tend, trendM, label='Línea de Tendencia', linestyle='--')

plt.axvline(522, color='red', linewidth=1, linestyle='dashed')
plt.axvline(632, color='red', linewidth=1, linestyle='dashed')
plt.axvline(795, color='red', linewidth=1, linestyle='dashed')
plt.axvline(1095, color='red', linewidth=1, linestyle='dashed')


plt.plot(t_tend, detrended_signal_M, label='Señal Sin Tendencia')
plt.legend()
plt.xlabel('Tiempo')

plt.ylabel('Amplitud')
plt.title('Lineas hombres')
plt.show()

#%%
def detectar_picos_y_modificar_senal(signal, umbral , retardo = 100):
    señal_modificada = np.zeros_like(signal)
    ventana = 90
    ultima_muestra = -retardo  # Inicializar con un valor negativo para evitar problemas en la primera iteración
    for i in range(ventana, len(signal) - ventana):
        if signal[i] > umbral and signal[i] > signal[i-1] and signal[i] > signal[i+1] and (i - ultima_muestra) > retardo:
            señal_modificada[i-ventana:i+ventana+1] = signal[i-ventana:i+ventana+1]
            ultima_muestra = i+ventana+1
    return señal_modificada
umbral = 0.01
# Modificar la señal
señal_modificada = detectar_picos_y_modificar_senal(detrended_signal_M, umbral)

# Graficar la señal original y la señal modificada
fig, (ax3, ax4) = plt.subplots(1, 2, figsize=(16, 8))
ax3.plot(t_tend,detrended_signal_M, label='Señal original')
ax3.axhline(y=umbral, color='g', linestyle='--', label='Umbral')
ax3.legend()

ax4.plot(t_tend, señal_modificada, label='Señal modificada', linestyle='-')
ax4.legend()


#%% Guardar promedios
# np.savetxt('promediofemenino.txt', y)
# np.savetxt('promediomasculino.txt', z)
