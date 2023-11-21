import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

# Para las matrices de 8 electrodos: 

"""
channels = ['Fz', 'C3', 'Cz', 'C4', 'Pz', 'PO7', 'Oz', 'PO8']

points3D = [[0, 0.71934, 0.694658], [-0.71934, 0, 0.694658], [0, 0, 1], [0.71934, 0, 0.694658], [0, -0.71934, 0.694658], [-0.587427, -0.808524, -0.0348995], [0, -0.999391, -0.0348995], [0.587427, -0.808524, -0.0348995]]
"""

# Para las matrcies de 32 electrodos:

#"""
channels = ['Fp1','Fp2', 'AF3', 'AF4', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'T7', 'C3', 'Cz', 'C4', 'T8', 'CP5', 'CP1', 'CP2', 'CP6', 'P7', 'P3', 'Pz', 'P4', 'P8', 'PO3', 'PO4', 'O1', 'Oz', 'O2']

points3D = [[-0.308829,0.950477,-0.0348995], [0.308829,0.950477,-0.0348995], [-0.406247,0.871199,0.275637], [0.406247,0.871199,0.275637], [-0.808524,0.587427,-0.0348995], [-0.545007,0.673028,0.5], [0,0.71934,0.694658], [0.545007,0.673028,0.5], [0.808524,0.587427,-0.0348995], [-0.887888,0.340828,0.309017], [-0.37471,0.37471,0.848048], [0.37471,0.37471,0.848048], [0.887888,0.340828,0.309017], [-0.999391,0,-0.0348995], [-0.71934,0,0.694658], [0,0,1], [0.71934,0,0.694658], [0.999391,0,-0.0348995], [-0.887888,-0.340828,0.309017], [-0.37471,-0.37471,0.848048], [0.37471,-0.37471, 0.848048], [0.887888,-0.340828,0.309017], [-0.808524,-0.587427,-0.0348995], [-0.545007,-0.673028,0.5], [0,-0.71934,0.694658], [0.545007,-0.673028,0.5], [0.808524,-0.587427,-0.0348995], [-0.406247,-0.871199,0.275637], [0.406247,-0.871199,0.275637], [-0.308829,-0.950477,-0.0348995], [0,-0.999391,-0.0348995], [0.308829,-0.950477,-0.0348995]]
#"""

points3D = np.array(points3D)

# Fórmulas para pasar de 3D a 2D
r = np.sqrt(points3D[:, 0]**2 + points3D[:, 1]**2 + points3D[:, 2]**2)
t = r / (r + points3D[:, 2])
x = t * points3D[:, 0]
y = t * points3D[:, 1]
points2D = np.column_stack((x, y))

def cargar_matriz(nombre_archivo):
    # Carga del archivo usando numpy.loadtxt
    matriz = np.loadtxt(nombre_archivo, skiprows=0, dtype=int)
    return matriz

def calcular_distancia(punto1, punto2):
    # Fórmula de distancia entre dos puntos en 3D
    return np.sqrt((punto1[0] - punto2[0])**2 + (punto1[1] - punto2[1])**2 + (punto1[2] - punto2[2])**2)

def calcular_casco_convexo_con_conexiones(puntos_2d, matriz):
    # Calcular el casco convexo para todos los puntos
    hull = ConvexHull(puntos_2d)
    
    # Obtener canales con al menos una conexión en la matriz
    canales_con_conexiones_filas = np.unique(np.where(matriz == 1)[0])
    canales_con_conexiones_columnas = np.unique(np.where(matriz == 1)[1])
    
    # Unir los conjuntos de canales con conexiones de filas y columnas
    canales_con_conexiones = np.union1d(canales_con_conexiones_filas, canales_con_conexiones_columnas)
    
    # Filtrar los vértices del casco convexo solo para los canales con conexiones
    puntos_filtrados = puntos_2d[canales_con_conexiones]
    
    # Calcular un nuevo casco convexo usando solo los puntos filtrados
    hull_filtrado = ConvexHull(puntos_filtrados)

    return hull_filtrado

def graficar_casco_convexo(ax, hull):
    for simplex in hull.simplices:
        ax.plot(hull.points[simplex, 0], hull.points[simplex, 1], 'r--', lw=2)

# Modificar la función graficar_conectividad para utilizar la nueva función
def graficar_conectividad(ax, matriz, canales, puntos_2d, puntos_3d):
    conexiones = np.argwhere(matriz == 1)
    ax.scatter(puntos_2d[:, 0], puntos_2d[:, 1])

    for conexion in conexiones:
        canal_origen = conexion[0]
        canal_destino = conexion[1]
        punto_origen = puntos_2d[canal_origen]
        punto_destino = puntos_2d[canal_destino]

        distancia = calcular_distancia(puntos_3d[canal_origen], puntos_3d[canal_destino])

        ax.plot([punto_origen[0], punto_destino[0]], [punto_origen[1], punto_destino[1]], 'k-', alpha=0.5)
        ax.text((punto_origen[0] + punto_destino[0]) / 2, (punto_origen[1] + punto_destino[1]) / 2, f'{distancia:.2f}', color='blue')

    hull = calcular_casco_convexo_con_conexiones(puntos_2d, matriz)
    graficar_casco_convexo(ax, hull)

    for i in range(len(puntos_2d)):
        ax.text(puntos_2d[i, 0] - 0.02, puntos_2d[i, 1] + 0.025, canales[i])

#### ARCHIVOS:
# S11 = Emilio Berber
#archivos = ["Lectura_s11.txt", "Memoria_s11.txt", "Operaciones_s11.txt"]
# S09 = Moisés Pineda
#archivos = ["Lectura_s09.txt", "Memoria_s09.txt", "Operaciones_s09.txt"]
# S07 = Samuel B
# archivos = ["Lectura_s07.txt", "Memoria_s07.txt", "Operaciones_s07.txt"]
# S0A = Matriz con 32 electrodos 
archivos = ["Lectura_s0a.txt", "Memoria_s0a.txt", "Operaciones_s0a.txt"]

fig, axs = plt.subplots(1, len(archivos), figsize=(15, 5))

# Modificar la parte del bucle para utilizar la nueva función
for ax, nombre_archivo in zip(axs, archivos):
    matriz = cargar_matriz(nombre_archivo)
    graficar_conectividad(ax, matriz, channels, points2D, points3D)
    circle = plt.Circle((0, 0), 1.04, color='r', alpha=0.25, fill=False)
    ax.add_patch(circle)

    # Utilizar la nueva función para calcular el casco convexo con conexiones específicas
    hull = calcular_casco_convexo_con_conexiones(points2D, matriz)
    graficar_casco_convexo(ax, hull)

    ax.set_title(nombre_archivo.replace(".txt", ""))
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.axis('equal')

plt.show()