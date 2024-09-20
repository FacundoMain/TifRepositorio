import os
import numpy as np
from scipy.signal import firwin, lfilter, butter, filtfilt, savgol_filter, find_peaks 
from scipy import fft,stats
import matplotlib.pyplot as plt
import math
import scipy.stats as stats
from scipy.interpolate import CubicSpline
from scipy.optimize import curve_fit

def detectar_picos(signal, umbral, retardo=100):
    '''
   Detecta picos en una señal dada que superan un umbral específico y tienen un retardo mínimo entre ellos.

    Parámetros:
    -----------
    signal : array_like
        La señal en la que se desean detectar los picos.
    umbral : float
        El valor umbral que un pico debe superar para ser considerado.
    retardo : int, opcional
        El número mínimo de muestras entre picos consecutivos. El valor por defecto es 100.

    Retorna:
    --------
    señal_modificada : ndarray
        Una copia de la señal original donde solo se conservan los segmentos alrededor de los picos detectados.
    indices_picos : list
        Una lista de índices donde se encuentran los picos detectados.

    '''
    señal_modificada = np.zeros_like(signal)
    indices_picos = []
    ultima_muestra = -retardo
    n = len(signal)

    for i in range(1, n - 1):
        if signal[i] > umbral and signal[i] > signal[i-1] and signal[i] > signal[i+1] and (i - ultima_muestra) > retardo:

            inicio = i
            while inicio > 0 and signal[inicio] * signal[inicio - 1] > 0:
                inicio -= 1


            fin = i
            while fin < n - 1 and signal[fin] * signal[fin + 1] > 0:
                fin += 1

            señal_modificada[inicio:fin+1] = signal[inicio:fin+1]
            indices_picos.append(i)  # Guardar el índice del pico máximo
            ultima_muestra = i

    return señal_modificada, indices_picos


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
    seniales_sin_filt = []
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
            
            #Almaceno las señales sin filtrar
            Senial = normalizador(samples)
            seniales_sin_filt.append((filename,Senial))
            
    #%% Señal levantada del txt
    T = 1/fs
    N = len(seniales_sin_filt[1][1])
    t = np.linspace(0, N*T, N, endpoint=False)
    
    filepath = os.path.join(folder, "006_1_23.txt")
    senal_levantada = load_samples(filepath)
    
    plt.figure(figsize=(16, 8))
    plt.plot(t, senal_levantada)
    plt.xlabel('Tiempo [s]')
    plt.ylabel('Amplitud[µS]')
    plt.title('Señal sin procesar: "006_1_23.txt"')
    plt.legend()
    plt.grid(True)
    plt.show()

    #%% Señal filtrada
    
    senal_filtrada = fir_lowpass_filter(remove_dc(senal_levantada))
    N = len(senal_filtrada)
    t = np.linspace(0, N*T, N, endpoint=False)
    
    plt.figure(figsize=(16, 8))
    plt.plot(t, senal_filtrada)
    plt.xlabel('Tiempo [s]')
    plt.ylabel('Amplitud[µS]')
    plt.title('Señal filtrada: "006_1_23.txt"')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    #%% Señal normalizada y filtrada
    senial_normalizada = normalizador(senal_filtrada)
    plt.figure(figsize=(16, 8))
    plt.plot(t, senial_normalizada)
    plt.xlabel('Tiempo [s]')
    plt.ylabel('Amplitud')
    plt.title('Señal filtrada y normalizada: "006_1_23"')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # la fig linda era la 034_1_23
    #%% Promedio y desvio en hombres y mujeres total
   
    T = 1/fs
    N = len(seniales_sin_filt[1][1])
    t = np.linspace(0, N*T, N, endpoint=False)
    
    # Sin filtrar
    promedioFemenino, promedioMasculino = promedioPorGenero(seniales_sin_filt)
    desvio_femenino, desvio_masculino = desvioPorGenero(seniales_sin_filt, promedioFemenino, promedioMasculino)
    
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle("Promedio normalizado y desvio estandar en todas las mediciones", fontsize = 18)
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
    
    
    #%% Filtradas
    
    T = 1/fs
    N = len(all_filtered_signals[1][1])
    t = np.linspace(0, N*T, N, endpoint=False)
    
    promedioFemenino, promedioMasculino = promedioPorGenero(all_filtered_signals)
    desvio_femenino, desvio_masculino = desvioPorGenero(all_filtered_signals, promedioFemenino, promedioMasculino)
    
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle("Promedio normalizado y desvio estandar en todas las mediciones", fontsize = 18)
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
    # ax1.plot(115, promedioMasculino[1140], 'ro', markersize=4, label='Picos')
    ax1.legend()
    ax1.grid(True)

    ax2.set_title("Promedio  femenino", fontsize = 15)
    ax2.plot(t,promedioFemenino,'r', label = 'Promedio')
    ax2.plot(t,(promedioFemenino + desvio_femenino), '--g', label = 'Desvio')
    ax2.plot(t,(promedioFemenino - desvio_femenino), '--g')
    ax2.set_xlabel('Tiempo [s]', fontsize = 12)
    ax2.set_ylabel('Amplitud [S]', fontsize = 12)
    ax2.set_xlim([0,400])
    # ax2.plot(115, promedioFemenino[1140], 'bo', markersize=4, label='Picos')
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
        

#%% Tramo 1
signalM = promedioMasculino[0:4000]
t_tend = t[0:4000]
umbral_1 = 0.01
trendM = savgol_filter(signalM, window_length=1000, polyorder=5)
slope, intercept, r_value, p_value, std_err = stats.linregress(t_tend,signalM)
detrended_signal_M = signalM- trendM
señal_modificada, picos_M = detectar_picos(detrended_signal_M, umbral_1)

fig, (ax3, ax4) = plt.subplots(1, 2, figsize=(16, 8))
# plt.figure(figsize=(10, 6))
ax3.set_title("Promedio masculino normalizado y picos detectados", fontsize = 15)
ax3.set_title("Promedio  masculino y picos detectados", fontsize = 15)
ax3.plot(t_tend, signalM,'b', label='Señal Original')
# plt.axhline(y=umbral, color='g', linestyle='--', label='Umbral')
ax3.plot(t_tend, señal_modificada,'#D95319', label=' Picos', linestyle='-')
ax3.set_xlabel('Tiempo [s]')
ax3.set_ylabel('Amplitud')
ax3.legend()
ax3.grid()

print("\n Tramo 1:")
print("\n Hombres")
for pico in picos_M:
    print(t_tend[pico],señal_modificada[pico])

signalF = promedioFemenino[0:4000]
t_tend = t[0:4000]

trendF = savgol_filter(signalF, window_length=1000, polyorder=5)
slope, intercept, r_value, p_value, std_err = stats.linregress(t_tend,signalF)
detrended_signal_F = signalF- trendF
señal_modificada, picos_F = detectar_picos(detrended_signal_F, 0.012)

ax4.set_title("Promedio femenino normalizado y picos detectados", fontsize = 15)
ax4.set_title("Promedio  femenino y picos detectados", fontsize = 15)
ax4.plot(t_tend, signalF,'r', label='Señal Original')
ax4.plot(t_tend, señal_modificada,'m', label='Picos', linestyle='-')
ax4.set_xlabel('Tiempo [s]')
ax4.set_ylabel('Amplitud')
ax4.legend()
ax4.grid()

print("\n Mujeres:") 
for pico in picos_F:
    print(t_tend[pico],señal_modificada[pico])
    
#%% Tramo 2
t_tend = t[4001:12250]

signalM = promedioMasculino[4001:12250]
trendM = savgol_filter(signalM, window_length=1000, polyorder=5)
slope, intercept, r_value, p_value, std_err = stats.linregress(t_tend,signalM)
detrended_signal_M = signalM- trendM

umbral = 0.006

señal_modificada_M, picos_M = detectar_picos(detrended_signal_M, umbral)

fig, (ax3, ax4) = plt.subplots(1, 2, figsize=(16, 8))
ax3.set_title("Promedio  masculino normalizado y picos detectados", fontsize = 15)
ax3.set_title("Promedio  masculino y picos detectados", fontsize = 15)
ax3.plot(t_tend, signalM,'b', label='Señal Original')
ax3.plot(t_tend, señal_modificada_M,'#D95319', label='Picos', linestyle='-')
ax3.set_xlabel('Tiempo [s]')
ax3.set_ylabel('Amplitud')
ax3.legend()
ax3.grid()

print("\n Tramo 2:")
print("\n Hombres")
for pico in picos_M:
    print(t_tend[pico],señal_modificada_M[pico])
    
    
signalF = promedioFemenino[4001:12250]
trendF = savgol_filter(signalF, window_length=1000, polyorder=5)
slope, intercept, r_value, p_value, std_err = stats.linregress(t_tend,signalF)
detrended_signal_F = signalF- trendF   
señal_modificada_F, picos_F = detectar_picos(detrended_signal_F, umbral)
ax4.set_title("Promedio  femenino normalizado y picos detectados", fontsize = 15)

ax4.set_title("Promedio  femenino y picos detectados", fontsize = 15)
ax4.plot(t_tend, signalF,'r', label='Señal Original')
ax4.plot(t_tend, señal_modificada_F,'m', label='Picos', linestyle='-')
ax4.set_xlabel('Tiempo [s]')
ax4.set_ylabel('Amplitud')
ax4.legend()
ax4.grid()



print("\n Mujeres:") 
for pico in picos_F:
    print(t_tend[pico],señal_modificada_F[pico])
    
    
#%% Tramo 3
fig, (ax3, ax4) = plt.subplots(1, 2, figsize=(16, 8))

signalM = promedioMasculino[12250:13140]
t_tend = t[12250:13140]
trendM = savgol_filter(signalM, window_length=100, polyorder=5)
slope, intercept, r_value, p_value, std_err = stats.linregress(t_tend,signalM)
detrended_signal_M = signalM- trendM
señal_modificada, picos_M = detectar_picos(detrended_signal_M, umbral)

ax3.set_title("Promedio masculino normalizado y picos detectados", fontsize = 15)
ax3.set_title("Promedio  masculino y picos detectados", fontsize = 15)
ax3.plot(t_tend, signalM,'b', label='Señal Original')
# plt.axhline(y=umbral, color='g', linestyle='--', label='Umbral')
ax3.plot(t_tend, señal_modificada,'#D95319', label=' Picos', linestyle='-')
ax3.set_xlabel('Tiempo [s]')
ax3.set_ylabel('Amplitud')
ax3.legend()
ax3.grid()

print("\n Tramo 3:")
print("\n Hombres:")
for pico in picos_M:
    print(t_tend[pico],señal_modificada[pico])

signalF = promedioFemenino[12250:13140]
t_tend = t[12250:13140]
trendF = savgol_filter(signalF, window_length=100, polyorder=5)
slope, intercept, r_value, p_value, std_err = stats.linregress(t_tend,signalF)
detrended_signal_F = signalF- trendF
señal_modificada, picos_F = detectar_picos(detrended_signal_F, umbral)


ax4.set_title("Promedio femenino normalizado y picos detectados", fontsize = 15)
ax4.set_title("Promedio  femenino y picos detectados", fontsize = 15)
ax4.plot(t_tend, signalF,'r', label='Señal Original')
ax4.plot(t_tend, señal_modificada,'m', label='Picos', linestyle='-')
ax4.set_xlabel('Tiempo [s]')
ax4.set_ylabel('Amplitud')
ax4.legend()
ax4.grid()


print("\n Mujeres:")    
for pico in picos_F:
    print(t_tend[pico],señal_modificada[pico])
