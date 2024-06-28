# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 21:49:28 2024

@author: nahue
"""
import numpy as np
import matplotlib.pyplot as plt

# Generar tres vectores con valores aleatorios
vector1 = np.random.rand(30)
vector2 = np.random.rand(30)
vector3 = np.random.rand(30)

# Calcular el promedio y la desviación estándar de cada vector
mean_vector1 = np.mean(vector1)
std_vector1 = np.std(vector1)

mean_vector2 = np.mean(vector2)
std_vector2 = np.std(vector2)

mean_vector3 = np.mean(vector3)
std_vector3 = np.std(vector3)

# Imprimir los resultados
print("Vector 1 - Promedio:", mean_vector1, "Desviación estándar:", std_vector1)
print("Vector 2 - Promedio:", mean_vector2, "Desviación estándar:", std_vector2)
print("Vector 3 - Promedio:", mean_vector3, "Desviación estándar:", std_vector3)

# Graficar los vectores
plt.figure(figsize=(12, 6))

plt.plot(vector1, marker='o', linestyle='-', color='b', label='Vector 1')
plt.plot(vector2, marker='s', linestyle='-', color='g', label='Vector 2')
plt.plot(vector3, marker='^', linestyle='-', color='r', label='Vector 3')

# Añadir promedios como líneas horizontales
plt.axhline(y=mean_vector1, color='b', linestyle='--', label='Promedio Vector 1')
plt.axhline(y=mean_vector2, color='g', linestyle='--', label='Promedio Vector 2')
plt.axhline(y=mean_vector3, color='r', linestyle='--', label='Promedio Vector 3')

# Personalizar el gráfico
plt.title('Vectores Aleatorios y sus Promedios')
plt.xlabel('Índice')
plt.ylabel('Valor')
plt.legend()
plt.grid(True)

# Mostrar el gráfico
plt.show()
