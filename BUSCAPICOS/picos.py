import pandas as pd
import numpy as np
from tabulate import tabulate  # Importar tabulate para formatear la tabla
from scipy.signal import find_peaks  # Importar find_peaks para encontrar picos

# Cargar el vector desde un archivo .npy
archivo_npy = 'promedioFemenido.npy'  # Ruta real de tu archivo .npy
vector = np.load(archivo_npy)

# Definir los límites de los fragmentos
fragmentos_limites = [
    (0, 950), (951, 1810), (1811, 2790), (2791, 3440),
    (3441, 3710), (3711, 4160), (4161, 4810), (4811, 5310),
    (5311, 5610), (5611, 5850), (5851, 6540), (6541, 7210),
    (7211, 7610), (7611, 8290), (8291, 8890), (8891, 9250),
    (9251, 9610), (9611, 10340), (10341, 10810), (10811, 11230),
    (11231, 11510), (11511, 12010), (12011, 13141)
]

# Convertir los límites de los fragmentos en índices
fragmentos_indices = [(inicio, fin) for inicio, fin in fragmentos_limites]

# Cargar los datos en un DataFrame de pandas
df = pd.DataFrame(vector, columns=['Valores'])

# Función para encontrar picos en un fragmento
def encontrar_picos(fragmento):
    indices_picos, _ = find_peaks(fragmento['Valores'], distance=10)  # Ajusta la distancia según tus necesidades
    picos = fragmento.iloc[indices_picos]
    return picos

# Encontrar picos en cada fragmento y guardar los resultados
resultados = []
for inicio, fin in fragmentos_indices:
    fragmento = df.iloc[inicio:fin]
    picos_fragmento = encontrar_picos(fragmento)
    resultados.append(picos_fragmento)

# Mostrar los resultados de los picos
for i, picos_fragmento in enumerate(resultados):
    inicio, fin = fragmentos_indices[i]
    print(f'Picos en Fragmento {i+1} ({inicio}-{fin}):')
    print(tabulate(picos_fragmento, headers='keys', tablefmt='pretty'))
    print()
