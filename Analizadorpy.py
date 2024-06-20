import os
import numpy as np
from scipy.signal import firwin, lfilter, butter, filtfilt
from scipy.fftpack import fft
import matplotlib.pyplot as plt
import math


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
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Amplitud')  
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
    return data

def derivative(data, fs=10):
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
    return d_data


def moving_average(data, window_size=400):
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
    
    if len(smoothed_data) == 12802:
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
    plt.ylabel('Amplitud')
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
        diff_cuad = [(senial[i] - promedio[i]) ** 2 for i in range(len(senial))]
        suma_diff_cuad = [suma_diff_cuad[i] + diff_cuad[i] for i in range(len(senial))]
    
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
            plot_time_signal(Senial, fs=fs, title=f"Señal Filtrada en el Tiempo - {filename}")
            #plot_frequency_spectrum(Senial)
        

    # Graficar todas las señales filtradas solapadas
    plt.figure(figsize=(12, 8))
    for filename, signal in all_filtered_signals:
        N = len(signal)
        T = 1/fs
        t = np.linspace(0, N*T, N, endpoint=False)
        plt.plot(t, signal)#, label=filename)
        
        
    plt.title("Todas las Señales Filtradas en el Tiempo")
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Amplitud')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    
    #%% no puedo aplicar la funcion pq no es una tupla ;)
    prom = np.zeros(N);
    
    for i in range(N):    
        for senial in seniales_sin_norm:
            prom[i] += senial[i] 
        prom[i]= prom[i]/len(seniales_sin_norm)
    
    # for i in range(N):
    #     prom[i]= prom[i]/len(seniales_sin_norm);
    
    # Inicializar la suma de diferencias al cuadrado
    suma_diff_cuad = [0] * len(seniales_sin_norm[0])
    
    # Sumar las diferencias al cuadrado para cada vector
    for senial in seniales_sin_norm:
        diff_cuad = [(senial[i] - prom[i]) ** 2 for i in range(len(senial))]
        suma_diff_cuad = [suma_diff_cuad[i] + diff_cuad[i] for i in range(len(senial))]
    
    # Calcular la varianza dividiendo por el número de vectores
    varianza = [suma_diff_cuad[i] / len(seniales_sin_norm) for i in range(len(suma_diff_cuad))]
    
    desvio_estandar = [math.sqrt(varianza[i]) for i in range(len(varianza))]
       
    
    VdesMax = prom+desvio_estandar
    VdesMin = prom-desvio_estandar
    
    plt.figure(figsize=(12, 8))
    
    plt.plot(t,prom,'g', label = 'Promedio')
    plt.plot(t,VdesMax, '--r', label = 'Desvio')
    plt.plot(t,VdesMin, '--r')
    
    plt.title("Promedio de todas las Señales")
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Amplitud')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    #%% Lo mismo que antes pero el promedio tiene las señales normalizadas
     
    promedio  = promediar(all_filtered_signals)
    desvio = calcularDesvio(all_filtered_signals, promedio)
    
    plt.figure(figsize=(12, 8))
    
    plt.plot(t,promedio,'g', label = 'Promedio')
    plt.plot(t,promedio+desvio, '--r', label = 'Desvio')
    plt.plot(t,promedio-desvio, '--r')
    
    plt.title("Promedio de todas las Señales normalizadas")
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Amplitud')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    
    
    
    #%% Junto las 2 en una figura para comparar
    # Crear un nuevo objeto Figure y dos Axes (subplots)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle("Promedio y desvio estandar en toda la población")
    ax1.plot(t,prom,'c', label = 'Promedio')
    ax1.plot(t,VdesMax, '--g', label = 'Desvio')
    ax1.plot(t,VdesMin, '--g')
    
    ax1.set_title("Promedio de todas las Señales")
    ax1.set_xlabel('Tiempo (s)')
    ax1.set_ylabel('Amplitud')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(t,promedio,'c', label = 'Promedio')
    ax2.plot(t,promedio+desvio, '--g', label = 'Desvio')
    ax2.plot(t,promedio-desvio, '--g')
    
    ax2.set_title("Promedio de todas las Señales normalizadas")
    ax2.set_xlabel('Tiempo (s)')
    ax2.set_ylabel('Amplitud')
    ax2.legend()
    ax2.grid(True)
    
    #%% Promedio y desvio en hombres y mujeres total
    promedioFemenino, promedioMasculino = promedioPorGenero(all_filtered_signals)
    desvio_femenino, desvio_masculino = desvioPorGenero(all_filtered_signals, promedioFemenino, promedioMasculino)
      
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle("Promedio y desvio estandar en todas las mediciones")
    ax1.set_title("Promedio  masculino")
    ax1.plot(t,promedioMasculino,'b', label = 'Promedio')
    ax1.plot(t,promedioMasculino + desvio_masculino, '--g', label = 'Desvio')
    ax1.plot(t,promedioMasculino - desvio_masculino, '--g')
    ax1.set_xlabel('Tiempo (s)')
    ax1.set_ylabel('Amplitud')
    ax1.legend()
    ax1.grid(True)
    
    ax2.set_title("Promedio  femenino")
    ax2.plot(t,promedioFemenino,'r', label = 'Promedio')
    ax2.plot(t,promedioFemenino + desvio_femenino, '--g', label = 'Desvio')
    ax2.plot(t,promedioFemenino - desvio_femenino, '--g')
    ax2.set_xlabel('Tiempo (s)')
    ax2.set_ylabel('Amplitud')
    ax2.legend()
    ax2.grid(True)
   
    #%%
    # Separo en secciones
    ruta_carpeta = 'RegistrosDia'
    seniales_diarias = []
    seniales_nocturnas = []
    
    for tupla in all_filtered_signals:
        nombre_archivo = tupla[0]
        ruta_completa = os.path.join(ruta_carpeta, nombre_archivo)
    
        if os.path.isfile(ruta_completa):
            seniales_diarias.append(tupla)
        
        else:
             seniales_nocturnas.append(tupla)

   # Dia y tardecita:
    promedio_dia = promediar(seniales_diarias)
    desvio_dia = calcularDesvio(seniales_diarias, promedio_dia)
    
    promedio_noche = promediar(seniales_nocturnas)
    desvio_noche = calcularDesvio(seniales_nocturnas, promedio_noche)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    plt.suptitle("Comparación entre mediciones")
    ax1.plot(t,promedio_dia,'c', label = 'Promedio')
    ax1.plot(t,promedio_dia + desvio_dia, '--g', label = 'Desvio')
    ax1.plot(t,promedio_dia - desvio_dia, '--g')
    
    ax1.set_title("Promedio de las Señales medidas a la mañana")
    ax1.set_xlabel('Tiempo (s)')
    ax1.set_ylabel('Amplitud')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(t,promedio_noche,'c', label = 'Promedio')
    ax2.plot(t,promedio_noche + desvio_noche, '--g', label = 'Desvio')
    ax2.plot(t,promedio_noche - desvio_noche, '--g')
    
    ax2.set_title("Promedio de las Señales medidas a la tarde")
    ax2.set_xlabel('Tiempo (s)')
    ax2.set_ylabel('Amplitud')
    ax2.legend()
    ax2.grid(True)
    
    #%% Mujeres vs hombres turno mñn
    promedioFemenino, promedioMasculino = promedioPorGenero(seniales_diarias)
    desvio_femenino, desvio_masculino = desvioPorGenero(seniales_diarias, promedioFemenino, promedioMasculino)
      
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle("Promedios por genero medidos a la mañana");
    ax1.set_title("Promedio  masculino")
    ax1.plot(t,promedioMasculino,'b', label = 'Promedio')
    ax1.plot(t,promedioMasculino + desvio_masculino, '--g', label = 'Desvio')
    ax1.plot(t,promedioMasculino - desvio_masculino, '--g')
    ax1.set_xlabel('Tiempo (s)')
    ax1.set_ylabel('Amplitud')
    ax1.legend()
    ax1.grid(True)
 
    ax2.set_title("Promedio  femenino")
    ax2.plot(t,promedioFemenino,'r', label = 'Promedio')
    ax2.plot(t,promedioFemenino + desvio_femenino, '--g', label = 'Desvio')
    ax2.plot(t,promedioFemenino - desvio_femenino, '--g')
    ax2.set_xlabel('Tiempo (s)')
    ax2.set_ylabel('Amplitud')
    ax2.legend()
    ax2.grid(True)
    
    #%% Mujeres vs hombres turno tarde
    promedioFemenino, promedioMasculino = promedioPorGenero(seniales_nocturnas)
    desvio_femenino, desvio_masculino = desvioPorGenero(seniales_nocturnas, promedioFemenino, promedioMasculino)
      
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle("Promedios por genero medidos a la tarde");
    ax1.set_title("Promedio  masculino")
    ax1.plot(t,promedioMasculino,'b', label = 'Promedio')
    ax1.plot(t,promedioMasculino + desvio_masculino, '--g', label = 'Desvio')
    ax1.plot(t,promedioMasculino - desvio_masculino, '--g')
    ax1.set_xlabel('Tiempo (s)')
    ax1.set_ylabel('Amplitud')
    ax1.legend()
    ax1.grid(True)
 
    ax2.set_title("Promedio  femenino")
    ax2.plot(t,promedioFemenino,'r', label = 'Promedio')
    ax2.plot(t,promedioFemenino + desvio_femenino, '--g', label = 'Desvio')
    ax2.plot(t,promedioFemenino - desvio_femenino, '--g')
    ax2.set_xlabel('Tiempo (s)')
    ax2.set_ylabel('Amplitud')
    ax2.legend()
    ax2.grid(True)