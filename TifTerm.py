import tkinter as tk
from tkinter import filedialog, messagebox
import serial
import serial.tools.list_ports
import time
import subprocess
import threading
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Variables globales para el manejo de datos y gráficos
max_data_points = 150
data = []

def listar_puertos_seriales():
    """Devuelve una lista de los puertos seriales disponibles."""
    puertos = serial.tools.list_ports.comports()
    return [puerto.device for puerto in puertos]

def actualizar_grafico(valor):
    """Actualiza el gráfico con nuevos datos."""
    data.append(valor)
    if len(data) > max_data_points:
        del data[0]

    # Actualiza los datos en la línea del gráfico
    lines.set_data(range(len(data)), data)

    # Ajusta los límites de los ejes
    ax.set_xlim(0, len(data))
    ax.set_ylim(min(data) - 10, max(data) + 10)

    # Redibuja el canvas
    canvas.draw()

def comenzar_monitoreo(puerto, ruta_archivo, archivo_mp3):
    """Comienza a monitorear el puerto serial y actualiza el osciloscopio."""
    ser = serial.Serial(puerto, 9600, timeout=1)
    ser.flush()
    inicio_tiempo = None
    diez_segundos_pasados = False

    try:
        with open(ruta_archivo, "w") as file:
            while True:
                if ser.in_waiting > 0:
                    linea = ser.readline().decode('utf-8').rstrip()
                    try:
                        valor = float(linea)  # Asegúrate de que los datos sean numéricos
                        actualizar_grafico(valor)
                        file.write(linea + "\n")
                        file.flush()
                    except ValueError:  # En caso de que la línea no sea un valor numérico
                        continue

                    if inicio_tiempo is None:
                        inicio_tiempo = time.time()

                    if not diez_segundos_pasados and (time.time() - inicio_tiempo >= 10):
                        diez_segundos_pasados = True
                        try:
                            subprocess.Popen(["cmd", "/c", archivo_mp3], shell=True)
                        except FileNotFoundError:
                            messagebox.showerror("Error", "No se pudo encontrar el archivo de audio.")
    except Exception as e:
        messagebox.showerror("Error", str(e))
    finally:
        ser.close()

def iniciar():
    puerto = puerto_var.get()
    ruta_archivo = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files", "*.txt")])
    if not ruta_archivo:
        messagebox.showinfo("Info", "Operación cancelada.")
        return
    
    archivo_mp3 = "Audio-de-Daniel-Zapata.mp3";
    
    if puerto and ruta_archivo and archivo_mp3:
        threading.Thread(target=comenzar_monitoreo, args=(puerto, ruta_archivo, archivo_mp3), daemon=True).start()
    else:
        messagebox.showinfo("Info", "Operación cancelada.")

root = tk.Tk()
root.title("Monitor Serial y Osciloscopio")

frame = tk.Frame(root)
frame.pack(padx=10, pady=10)

puertos = listar_puertos_seriales()
puerto_var = tk.StringVar(root)
puerto_var.set(puertos[0] if puertos else "")
combo_puertos = tk.OptionMenu(frame, puerto_var, *puertos)
combo_puertos.pack(pady=5, padx=(0,15))

boton_iniciar = tk.Button(frame, text="Iniciar", command=iniciar)
boton_iniciar.pack(side=tk.LEFT, padx=(0, 5))

# Configuración del gráfico
fig, ax = plt.subplots()
lines, = ax.plot([], [], 'r-')  # Inicializa una línea vacía
ax.set_xlim(0, max_data_points)  # Limita los datos mostrados en el eje x a 150
ax.set_ylim(0, 1023)  # Ajusta este límite según tus datos

canvas = FigureCanvasTkAgg(fig, master=root)  # Crea el canvas de matplotlib en Tkinter
widget = canvas.get_tk_widget()
widget.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

root.mainloop()
