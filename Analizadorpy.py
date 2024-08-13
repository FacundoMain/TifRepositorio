import os
import numpy as np
from scipy.signal import firwin, lfilter, butter, filtfilt, savgol_filter, find_peaks 
from scipy import fft,stats
import matplotlib.pyplot as plt
import math
import scipy.stats as stats
from scipy.interpolate import CubicSpline
from scipy.optimize import curve_fit

def fft_mag(x, fs = 10):
    """
    ------------------------
    INPUT:
    --------
    x: array de una dimensión conteniendo la señal cuya fft se busca calcular
    fs: frecuncia a la que está muestreada la señal
    ------------------------
    OUTPUT:
    --------
    f: array de una dimension con con los valores correspondientes al eje de 
    frecuencias de la fft.
    mag: array de una dimensión conteniendo los valores en magnitud de la fft
    de la señal.    
    """
    freq = fft.fftfreq(len(x), d=1/fs)   # se genera el vector de frecuencias
    senial_fft = fft.fft(x)    # se calcula la transformada rápida de Fourier

    # El espectro es simétrico, nos quedamos solo con el semieje positivo
    f = freq[np.where(freq >= 0)]      
    senial_fft = senial_fft[np.where(freq >= 0)]

    # Se calcula la magnitud del espectro
    mag = np.abs(senial_fft) / len(x)    # Respetando la relación de Parceval
    # Al haberse descartado la mitad del espectro, para conservar la energía 
    # original de la señal, se debe multiplicar la mitad restante por dos (excepto
    # en 0 y fm/2)
    mag[1:len(mag)-1] = 2 * mag[1:len(mag)-1]
    
    return f, mag

def plot_time_signal(data, fs=10, title="Señal en el Tiempo"):
    '''
    Grafica una señal en el dominio del tiempo.

    Parámetros
    ----------
    data : array-like
        Vector que almacena la magnitud a graficar.
    fs : float, opcional
        Frecuencia de muestreo. El valor predeterminado es 10.
    title : str, opcional
        Título que se desea colocar en la gráfica. Por defecto es "Señal en el Tiempo".
    '''
    N = len(data)  # Número total de muestras
    T = 1/fs  # Periodo de muestreo
    t = np.linspace(0, N*T, N, endpoint=False)  # Vector de tiempo
    plt.figure(figsize=(10, 6))
    plt.plot(t, data, label='Señal x[n]')
    plt.title(title)
    plt.xlabel('Tiempo [s]')
    plt.ylabel('Amplitud [Ω]')  
    plt.grid(True) 
    plt.legend()
    plt.show()

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

def butter_lowpass_filter(data, cutoff= 0.2, fs=10, order=10):
    '''
    Aplica un filtro pasabajos Butterworth a una señal en el dominio del tiempo.

    Parámetros
    ----------
    data : array-like
        Vector que almacena la magnitud de la señal a filtrar.
    
    cutoff : float, opcional
        Frecuencia de corte del filtro (normalizada respecto a la frecuencia de Nyquist). El valor predeterminado es 0.2.
    
    fs : float, opcional
        Frecuencia de muestreo. El valor predeterminado es 10.
   
    order : int, opcional
        Orden del filtro Butterworth. El valor predeterminado es 10.

    Returns
    -------
    senial: array-like
        Vector con la señal filtrada.
    '''
    nyq = 0.5 * fs  # Frecuencia de Nyquist
    normal_cutoff = cutoff / nyq
    # Obtiene los coeficientes del filtro
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    # Aplica el filtro
    senial = filtfilt(b, a, data)
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

def derivative(data, fs=10, threshold = 0.005):
    '''
    Calcula la derivada numérica de una señal en el dominio del tiempo.

    Parámetros
    ----------
    data : array-like
        Vector que almacena la magnitud de la señal.
    
    fs : float, opcional
        Frecuencia de muestreo. El valor predeterminado es 10.

    Returns
    -------
    d_data : array-like
        Vector con la derivada calculada.
    '''
    dt = 1/fs
    # Diferencia centrada excepto para los bordes
    d_data = np.diff(data) / dt
    
    # Ajustar valores de la derivada en el umbral
    d_data_adjusted = np.copy(d_data)
    d_data_adjusted[np.abs(d_data) < threshold] = 0
    
    d_data_adjusted[0:30] = d_data_adjusted[35]
    
    
    return d_data#d_data_adjusted

def adaptive_threshold(signal, window_size = 20, k =  1.35):
    
    # window_size =Tamaño de la ventana para el cálculo del umbral
    # K = Factor de escalado para el umbral adaptativo
    
    thresholds = np.zeros(len(signal))
    for i in range(window_size, len(signal)):
        window = signal[i-window_size:i]
        mean = np.mean(window)
        std_dev = np.std(window)
        thresholds[i] = mean + k * std_dev
    return thresholds


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

    
def plot_frequency_spectrum(data, fs=10, title="Espectro de Frecuencia"):
    '''
    Grafica el espectro de frecuencia de una señal en el dominio del tiempo.

    Parámetros
    ----------
    data : array-like
        Vector que almacena la magnitud de la señal.
    fs : float, opcional
        Frecuencia de muestreo. El valor predeterminado es 10.
    title : str, opcional
        Título que se desea colocar en la gráfica. Por defecto es "Espectro de Frecuencia".

    Returns
    -------
    None
    '''
    N = len(data)
    # Calcula la Transformada de Fourier
    yf = fft(data)
    xf = np.linspace(0.0, fs/2, N//2)
    plt.figure(figsize=(10, 6))
    plt.plot(xf, 2.0/N * np.abs(yf[:N//2]))
    plt.title(title)
    plt.xlabel('Frecuencia (Hz)')
    plt.ylabel('Amplitud [Ω]')
    plt.xlim([0, 0.1])
    plt.grid(True)
    plt.show()

def promediar(all_filtered_signals):
    '''
    Calcula el promedio de un conjunto de señales normalizadas.

    Parámetros
    ----------
    all_filtered_signals : list of tuples
        Lista de tuplas donde cada tupla contiene (nombre, señal normalizada).

    Returns
    -------
    promN : array-like
        Vector con el promedio de las señales.
    '''
    seniales_norm = [row[1] for row in all_filtered_signals]
    N = len(seniales_norm[0])
    promN = np.zeros(N)
    
    for i in range(N):    
        for senial in seniales_norm:
            promN[i] += senial[i] 
        promN[i] = promN[i] / len(seniales_norm)
    
    return promN

def calcularDesvio(all_filtered_signals, promedio):
    '''
    Calcula el desvío estándar de un conjunto de señales normalizadas.

    Parámetros
    ----------
    all_filtered_signals : list of tuples
        Lista de tuplas donde cada tupla contiene (nombre, señal normalizada).
    promedio : array-like
        Vector con el promedio de las señales.

    Returns
    -------
    desvio_estandar : array-like
        Vector con el desvío estándar de las señales.
    '''
    seniales_norm = [row[1] for row in all_filtered_signals]
    
    # Inicializar la suma de diferencias al cuadrado
    suma_diff_cuad = [0] * len(seniales_norm[0])
    
    # Sumar las diferencias al cuadrado para cada vector
    for senial in seniales_norm:
        diff_cuad = [(senial[i] - promedio[i]) ** 2 for i in range(len(Senial))]
        suma_diff_cuad = [suma_diff_cuad[i] + diff_cuad[i] for i in range(len(Senial))]
    
    # Calcular la varianza dividiendo por el número de vectores
    varianza = [suma_diff_cuad[i] / len(seniales_sin_norm) for i in range(len(suma_diff_cuad))]
    
    desvio_estandar = [math.sqrt(varianza[i]) for i in range(len(varianza))]
    
    return desvio_estandar

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


def desvioCorregido(desvio):
    '''
    Corrige un vector que sea la suma/resta del promedio y el desvío estandar para que valor sea solo entre 0 y 1.

    ----------
    desvio : array-like
        Vector a normalizar.

    Returns
    -------
    desvio : array-like
        Devuelve el vector que contiene solo valores entre 0 y 1.

    '''
    desvio = desvio #np.clip(desvio,0,1)
        
    return desvio
    
    

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
            #filtered_samples = butter_lowpass_filter(samples_no_dc, fs=fs)
            
            # Guardo señales sin normalizar
            
            seniales_sin_norm.append((filtered_samples))
            
            # window_size = 400
            # smoothed_data = moving_average(filtered_samples, window_size)
            
            Senial = normalizador(filtered_samples)
            
            # Almacenar la señal filtrada
            all_filtered_signals.append((filename, Senial))
            
            # Graficar la señal en el tiempo
            # plot_time_signal(Senial, fs=fs, title=f"Señal Filtrada en el Tiempo - {filename}")
                    

    # Graficar todas las señales filtradas solapadas
    # plt.figure(figsize=(12, 8))
    # for filename, signal in all_filtered_signals:
    #     N = len(signal)
    #     T = 1/fs
    #     t = np.linspace(0, N*T, N, endpoint=False)
    #     plt.plot(t, signal)#, label=filename)
        
        
    # plt.title("Todas las Señales Filtradas en el Tiempo")
    # plt.xlabel('Tiempo [s]')
    # plt.ylabel('Amplitud [Ω]')
    # # plt.legend()
    # plt.grid(True)
    # plt.show()
    
    
    #%% Lo mismo que antes pero el promedio tiene las señales normalizadas
    
    T = 1/fs
    N = len(all_filtered_signals[1][1])
    t = np.linspace(0, N*T, N, endpoint=False)
    
    promedio  = promediar(all_filtered_signals)
    desvio = calcularDesvio(all_filtered_signals, promedio)
    
    # plt.figure(figsize=(12, 8))
    
    # plt.plot(t,promedio,'black', label = 'Promedio')
    # plt.plot(t,desvioCorregido(promedio+desvio), '--g', label = 'Desvio')
    # plt.plot(t,desvioCorregido(promedio-desvio), '--g')
    
    # plt.title("Promedio de todas las Señales normalizadas")
    # plt.xlabel('Tiempo [s]')
    # plt.ylabel('Amplitud [uS]')
    # plt.xlim([0,len(t)/10])
    
    # # plt.axvline(100, color='c', linestyle=(0, (5, 10)), linewidth=1, label='Inicia el audio')
    # # plt.axvline(1250, color="#D95319", linestyle=(0, (5, 10)), linewidth=1, label='Finaliza el audio')


    # plt.legend()
    # plt.grid(True)
    # plt.show()
    
    # #%%
    derivTotal = derivative(promedio)
    tderiv = t[0:len(t)-1]
    
    # plt.figure(figsize=(12, 8))
    # plt.plot(tderiv,derivTotal,'black', label = 'Promedio')
    # plt.title("Derivada del promedio de todas las Señales")
    # plt.xlabel('Tiempo [s]')
    # plt.ylabel('Amplitud [S/s]')
    # plt.xlim([0,len(tderiv)/10])
    # plt.grid(True)
    # plt.show()
  
    # #%%
    f, E_total = fft_mag(derivTotal)
   
    # plt.figure(figsize=(12, 8))
    # plt.plot(f, E_total, 'black', label = 'Promedio')
    # plt.title("Espectro de la derivada del promedio")
    # plt.xlabel('Frecuencia [Hz]')
    # plt.ylabel('Amplitud [S/s]')
    # plt.xlim([0 ,0.1])
    # plt.grid(True)
    # plt.show()
        
    #%% Promedio y desvio en hombres y mujeres total
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
    ax1.set_xlim([0,len(tderiv)/10])
    ax1.legend()
    ax1.grid(True)

    ax2.set_title("Promedio  femenino", fontsize = 15)
    ax2.plot(t,promedioFemenino,'r', label = 'Promedio')
    ax2.plot(t,desvioCorregido(promedioFemenino + desvio_femenino), '--g', label = 'Desvio')
    ax2.plot(t,desvioCorregido(promedioFemenino - desvio_femenino), '--g')
    ax2.set_xlabel('Tiempo [s]', fontsize = 12)
    ax2.set_ylabel('Amplitud', fontsize = 12)
    ax2.set_xlim([0,len(tderiv)/10])
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
    ax2.plot(t,desvioCorregido(promedioFemenino + desvio_femenino), '--g', label = 'Desvio')
    ax2.plot(t,desvioCorregido(promedioFemenino - desvio_femenino), '--g')
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
    ax2.plot(t,desvioCorregido(promedioFemenino + desvio_femenino), '--g', label = 'Desvio')
    ax2.plot(t,desvioCorregido(promedioFemenino - desvio_femenino), '--g')
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
    ax2.plot(t,desvioCorregido(promedioFemenino + desvio_femenino), '--g', label = 'Desvio')
    ax2.plot(t,desvioCorregido(promedioFemenino - desvio_femenino), '--g')
    ax2.set_xlabel('Tiempo [s]', fontsize = 12)
    ax2.set_ylabel('Amplitud', fontsize = 12)
    ax2.set_xlim([1226,len(t)/10])
    ax2.legend()
    ax2.grid(True)
    
    #%%
    promedioFemenino, promedioMasculino = promedioPorGenero(all_filtered_signals)
    derivFem = derivative(promedioFemenino)
    derivMas = derivative(promedioMasculino)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    fig.suptitle("Derivada del promedio por generos", fontsize = 20);
    ax1.set_title("Masculino", fontsize = 15)
    ax1.plot(tderiv,derivMas,'b') 
    ax1.set_xlabel('Tiempo [s]')
    ax1.set_ylabel('Amplitud [S/s]', fontsize = 12)
    #ax1.set_ylim([-0.005,0.008])
    ax1.set_xlim([0,len(tderiv)/10])
    ax1.grid(True)
 
    ax2.set_title("Femenino", fontsize = 15)
    ax2.plot(tderiv,derivFem,'r') 
    ax2.set_xlabel('Tiempo [s]')
    ax2.set_ylabel('Amplitud [S/s]', fontsize = 12)
    #ax2.set_ylim([-0.005,0.008])
    ax2.set_xlim([0,len(tderiv)/10])
    ax2.grid(True)
    
    
    
    #%% Espectro derivada Generos
    
    f, E_Masculino = fft_mag(derivMas)
   
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    fig.suptitle("Espectro de la derivada del promedio por generos", fontsize = 20);
    ax1.set_title("Masculino", fontsize = 15)
    ax1.plot(f,E_Masculino,'b') 
    ax1.set_xlabel('Frecuencia [Hz]')
    ax1.set_ylabel('Amplitud [S/s]', fontsize = 12)
    ax1.set_xlim([0 ,1])
    ax1.grid(True)
    
    f, E_Femenino = fft_mag(derivFem)
    ax2.set_title("Femenino", fontsize = 15)
    ax2.plot(f,E_Femenino,'r') 
    ax2.set_xlabel('Frecuencia [Hz]')
    ax2.set_ylabel('Amplitud [S/s]', fontsize = 12)
    ax2.set_xlim([0 ,1])
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
    ax4.plot(t,desvioCorregido(promedioFemenino + desvio_femenino), '--g', label = 'Desvio')
    ax4.plot(t,desvioCorregido(promedioFemenino - desvio_femenino), '--g')
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
    
#%% Eleccion de prueba
# Existe distribucion normal?

# Test de Shapiro-Wilk para hombres
stat_hombres, p_value_hombres = stats.shapiro(promedioMasculino)
print('Shapiro-Wilk Test para hombres: estadístico =', stat_hombres, ', p-valor =', p_value_hombres)

# Test de Shapiro-Wilk para mujeres
stat_mujeres, p_value_mujeres = stats.shapiro(promedioFemenino)
print('Shapiro-Wilk Test para mujeres: estadístico =', stat_mujeres, ', p-valor =', p_value_mujeres)

# Comparo varianzas
# Prueba de Levene
stat_levene, p_value_levene = stats.levene(promedioMasculino, promedioFemenino)
print(f'Prueba de Levene: estadístico = {stat_levene}, p-valor = {p_value_levene}')


#%% PRUEBA DE SIGNIFICANCIA
t_stat, p_value = stats.ttest_ind(promedioMasculino[0:4000], promedioFemenino[0:4000],equal_var = False)

# Imprime los resultados
print(f"Estadístico t: {t_stat} en primer segmento")
print(f"Valor p: {p_value} en primer segmento")

# Decide si rechazar o no la hipótesis nula
alpha = 0.05
if p_value < alpha:
    print("PRIMER SEGMENTO: Rechazamos la hipótesis nula: Hay una diferencia significativa en la respuesta GSR entre hombres y mujeres.")
else:
    print("PRIMER SEGMENTO: No rechazamos la hipótesis nula: No hay una diferencia significativa en la respuesta GSR entre hombres y mujeres.")
    
    
t_stat, p_value = stats.ttest_ind(promedioMasculino[4001:12250], promedioFemenino[4001:12250],equal_var = False)
# Imprime los resultados
print(f"Estadístico t: {t_stat} en segundo segmento")
print(f"Valor p: {p_value} en segundo segmento")

    # Decide si rechazar o no la hipótesis nula
alpha = 0.05
if p_value < alpha:
    print("SEGUNDO SEGMENTO: Rechazamos la hipótesis nula: Hay una diferencia significativa en la respuesta GSR entre hombres y mujeres.")
else:
    print("SEGUNDO SEGMENTO: No rechazamos la hipótesis nula: No hay una diferencia significativa en la respuesta GSR entre hombres y mujeres.")
t_stat, p_value = stats.ttest_ind(promedioMasculino[12251:13141], promedioFemenino[12251:13141],equal_var = False)
# Imprime los resultados
print(f"Estadístico t: {t_stat} en tercer segmento")
print(f"Valor p: {p_value} en tercer segmento")

    # Decide si rechazar o no la hipótesis nula
alpha = 0.05
if p_value < alpha:
    print("TERCER SEGMENTO: Rechazamos la hipótesis nula: Hay una diferencia significativa en la respuesta GSR entre hombres y mujeres.")
else:
    print("TERCER SEGMENTO: No rechazamos la hipótesis nula: No hay una diferencia significativa en la respuesta GSR entre hombres y mujeres.")
    
#%% ANalisis intervalos para matriz de correlacion:
# Dividir los datos en segmentos
segmento1 = promedio[0:4000]
segmento2 = promedio[4001:12250]
segmento3 = promedio[12251:13141]
# Prueba de Levene para igualdad de varianzas
levene_stat12, levene_p12 = stats.levene(segmento1, segmento2)
levene_stat13, levene_p13 = stats.levene(segmento1, segmento3)
levene_stat23, levene_p23 = stats.levene(segmento2, segmento3)

# Determinar si usar equal_var=True o False en función del resultado de Levene
equal_var_12 = False
equal_var_13 = False
equal_var_23 = False

# Realizar pruebas t de Student
t_stat12, p_value12 = stats.ttest_ind(segmento1, segmento2, equal_var=equal_var_12)
t_stat13, p_value13 = stats.ttest_ind(segmento1, segmento3, equal_var=equal_var_13)
t_stat23, p_value23 = stats.ttest_ind(segmento2, segmento3, equal_var=equal_var_23)

# Crear la tabla de resultados
comparison_table_corrected = {
    "Segmento 1": ["1", f"{p_value12:.6f}", f": {p_value13:.6f}"],
    "Segmento 2": [f": {p_value12:.6f}", "1", f": {p_value23:.6f}"],
    "Segmento 3": [f" {p_value13:.6f}", f"{p_value23:.6f}", "1"]
}
comparison_table_corrected

comparison_table_corrected

#%% Comparo entre hombres y mujeres antes el mismo estimulo:

equal_var = False    

# 95-180		Inicio del relato y ambientacion de la persona
Masculino = promedioMasculino[950:1800]
Femenino = promedioFemenino[950:1800]

tstats, p_value = stats.ttest_ind(Masculino, Femenino, equal_var=equal_var)
   
print(f"Valor p: {p_value} en la respuesta del estimulo entre 95-180")

    # Decide si rechazar o no la hipótesis nula
alpha = 0.001
if p_value < alpha:
    print("Rechazamos la hipótesis nula: Hay una diferencia significativa en la respuesta GSR entre hombres y mujeres.\n")
else:
    print("No rechazamos la hipótesis nula: No hay una diferencia significativa en la respuesta GSR entre hombres y mujeres.\n")
   
    
   
#%%
#481-530		Estrella "ilumina todo el craneo" y luego comienza a descender hacia la garganta
Masculino = promedioMasculino[4810:5300]
Femenino = promedioFemenino[4810:5300]

tstats, p_value = stats.ttest_ind(Masculino, Femenino, equal_var=equal_var)
   
print(f"Valor p: {p_value} en la respuesta del estimulo entre 481-530")

    # Decide si rechazar o no la hipótesis nula
alpha = 0.001
if p_value < alpha:
    print("Rechazamos la hipótesis nula: Hay una diferencia significativa en la respuesta GSR entre hombres y mujeres.\n")
else:
    print("No rechazamos la hipótesis nula: No hay una diferencia significativa en la respuesta GSR entre hombres y mujeres.\n")
     
#585-653		Estrella se expande y ocupa todo el torax, expandiendose hacia el abdomen
Masculino = promedioMasculino[5850:6530]
Femenino = promedioFemenino[5850:6530]

tstats, p_value = stats.ttest_ind(Masculino, Femenino, equal_var=equal_var)
   
print(f"Valor p: {p_value} en la respuesta del estimulo entre 585-653")

    # Decide si rechazar o no la hipótesis nula
alpha = 0.001
if p_value < alpha:
    print("Rechazamos la hipótesis nula: Hay una diferencia significativa en la respuesta GSR entre hombres y mujeres.\n")
else:
    print("No rechazamos la hipótesis nula: No hay una diferencia significativa en la respuesta GSR entre hombres y mujeres.\n")

# 761-828		La Luz busca atraviesa la piel, el cuerpo se sumerje en una atmosfera luminosa que se expande
Masculino = promedioMasculino[7610:8280]
Femenino = promedioFemenino[7610:8280]

tstats, p_value = stats.ttest_ind(Masculino, Femenino, equal_var=equal_var)
   
print(f"Valor p: {p_value} en la respuesta del estimulo entre 761-828")

    # Decide si rechazar o no la hipótesis nula
alpha = 0.001
if p_value < alpha:
    print("Rechazamos la hipótesis nula: Hay una diferencia significativa en la respuesta GSR entre hombres y mujeres.\n")
else:
    print("No rechazamos la hipótesis nula: No hay una diferencia significativa en la respuesta GSR entre hombres y mujeres.\n")

#925-960		Cuerpo entero "brillando" y la luz ejerce presion en la piel, entrando al cuerpo
Masculino = promedioMasculino[9250:9600]
Femenino = promedioFemenino[9250:9600]

tstats, p_value = stats.ttest_ind(Masculino, Femenino, equal_var=equal_var)
   
print(f"Valor p: {p_value} en la respuesta del estimulo entre 925-960")

    # Decide si rechazar o no la hipótesis nula
alpha = 0.001
if p_value < alpha:
    print("Rechazamos la hipótesis nula: Hay una diferencia significativa en la respuesta GSR entre hombres y mujeres.\n")
else:
    print("No rechazamos la hipótesis nula: No hay una diferencia significativa en la respuesta GSR entre hombres y mujeres.\n")


#961-1033	La luz vuelve desde las extremidades, subiendo por el cuerpo. La persona "siente enorme bienestar"
Masculino = promedioMasculino[9610:10330]
Femenino = promedioFemenino[9610:10330]

tstats, p_value = stats.ttest_ind(Masculino, Femenino, equal_var=equal_var)
   
print(f"Valor p: {p_value} en la respuesta del estimulo entre 961-1033")

    # Decide si rechazar o no la hipótesis nula
alpha = 0.001
if p_value < alpha:
    print("Rechazamos la hipótesis nula: Hay una diferencia significativa en la respuesta GSR entre hombres y mujeres.\n")
else:
    print("No rechazamos la hipótesis nula: No hay una diferencia significativa en la respuesta GSR entre hombres y mujeres.\n")

#1081-1122	Estrella comienza a subir hacia la cabeza, pasando por la garganta. Se asienta en el centro de la cabeza
Masculino = promedioMasculino[10810:11220]
Femenino = promedioFemenino[10810:11220]

tstats, p_value = stats.ttest_ind(Masculino, Femenino, equal_var=equal_var)
   
print(f"Valor p: {p_value} en la respuesta del estimulo entre 1081-1122")

    # Decide si rechazar o no la hipótesis nula
alpha = 0.001
if p_value < alpha:
    print("Rechazamos la hipótesis nula: Hay una diferencia significativa en la respuesta GSR entre hombres y mujeres.\n")
else:
    print("No rechazamos la hipótesis nula: No hay una diferencia significativa en la respuesta GSR entre hombres y mujeres.\n")

#%% Busco relacion entre 1100s y 510s
segmento1 = promedio[4810:5300]
segmento2 = promedio[10810:11220]

equal_var_12 = False 

# Realizar pruebas t de Student
t_stat12, p_value12 = stats.ttest_ind(segmento1, segmento2, equal_var=equal_var_12)

print(f"Valor p: {p_value12} en la respuesta del estimulo de 1100s y 510s")

    # Decide si rechazar o no la hipótesis nula
alpha = 0.001
if p_value12 < alpha:
    print("Rechazamos la hipótesis nula: Hay una diferencia significativa en la respuesta GSR entre hombres y mujeres.\n")
else:
    print("No rechazamos la hipótesis nula: No hay una diferencia significativa en la respuesta GSR entre hombres y mujeres.\n")


#%% Funcion matematica:
plt.figure()
limInf = 10950 #10810
limSup = 11180 #11220


# Datos de ejemplo (puedes reemplazar estos con tus propios datos)
x = t[limInf:11030]#:limSup]
y = promedioFemenino[limInf:11030]#limSup]

plt.plot(x,y)
plt.grid()

#%%

# # Definición de la función exponencial
# def exponential_func(x, a, b):
#     return a * np.exp(b * x)

# # Ajuste de la función exponencial a los datos
# popt, pcov = curve_fit(exponential_func, x, y)

# # Parámetros ajustados
# a_fit, b_fit = popt

# # Valores ajustados
# y_fit = exponential_func(x, a_fit, b_fit)

# # Gráfico
# plt.scatter(x, y, label='Datos Originales')
# plt.plot(x, y_fit, label='Exponencial Ajustada', color='red')
# plt.legend()
# plt.show()



#%% Guardar promedios
# np.savetxt('promediofemenino.txt', promedioFemenino)
# np.savetxt('promediomasculino.txt', promedioMasculino)

