## ETAPA 5
"""
Un análisis típico que se hace en conectividad funcional es contar cuantas aristas están conectados a cada vértice. En esta etapa utilizarás diagramas de Voronoi para representar dichos valores Para ello:

1. Calcula el grado de cada vértice de tus grafos no conectados (número de aristas de cada vértice).
2. Para cada grafo, construye un diagrama de Voronoi en el que los puntos los determinan las posiciones de los electrodos en 2D.
3. Pinta cada región de los diagramas con un color que represente el número de aristas conectadas a cada vértice. Para asignar colores, utiliza una escada de colores adecuada.
4. Si puedes, dibuja un círculo de radio 1 centrado en el origen, el cual representa la cabeza.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, ConvexHull, Delaunay, voronoi_plot_2d # Librería para importar el algorimto de VORONOI y TRIANGULACIÓN DE DELUNAY

# Para las matrices de 8 electrodos: 

#"""
channels = ['Fz', 'C3', 'Cz', 'C4', 'Pz', 'PO7', 'Oz', 'PO8']

points3D = [[0, 0.71934, 0.694658], [-0.71934, 0, 0.694658], [0, 0, 1], [0.71934, 0, 0.694658], [0, -0.71934, 0.694658], [-0.587427, -0.808524, -0.0348995], [0, -0.999391, -0.0348995], [0.587427, -0.808524, -0.0348995]]
#"""

# Para las matrcies de 32 electrodos:

"""
channels = ['Fp1','Fp2', 'AF3', 'AF4', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'T7', 'C3', 'Cz', 'C4', 'T8', 'CP5', 'CP1', 'CP2', 'CP6', 'P7', 'P3', 'Pz', 'P4', 'P8', 'PO3', 'PO4', 'O1', 'Oz', 'O2']

points3D = [[-0.308829,0.950477,-0.0348995], [0.308829,0.950477,-0.0348995], [-0.406247,0.871199,0.275637], [0.406247,0.871199,0.275637], [-0.808524,0.587427,-0.0348995], [-0.545007,0.673028,0.5], [0,0.71934,0.694658], [0.545007,0.673028,0.5], [0.808524,0.587427,-0.0348995], [-0.887888,0.340828,0.309017], [-0.37471,0.37471,0.848048], [0.37471,0.37471,0.848048], [0.887888,0.340828,0.309017], [-0.999391,0,-0.0348995], [-0.71934,0,0.694658], [0,0,1], [0.71934,0,0.694658], [0.999391,0,-0.0348995], [-0.887888,-0.340828,0.309017], [-0.37471,-0.37471,0.848048], [0.37471,-0.37471, 0.848048], [0.887888,-0.340828,0.309017], [-0.808524,-0.587427,-0.0348995], [-0.545007,-0.673028,0.5], [0,-0.71934,0.694658], [0.545007,-0.673028,0.5], [0.808524,-0.587427,-0.0348995], [-0.406247,-0.871199,0.275637], [0.406247,-0.871199,0.275637], [-0.308829,-0.950477,-0.0348995], [0,-0.999391,-0.0348995], [0.308829,-0.950477,-0.0348995]]
"""

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

def graficar_conectividad(ax, matriz, canales, puntos_2d, puntos_3d):
    # Obtener índices de canales conectados
    conexiones = np.argwhere(matriz == 1)

    # Graficar puntos
    ax.scatter(puntos_2d[:, 0], puntos_2d[:, 1])

    # Etiquetar los puntos
    for i in range(len(puntos_2d)):
        ax.text(puntos_2d[i, 0] - 0.02, puntos_2d[i, 1] + 0.025, canales[i])


# Función para calcular el grado de cada vértice
def calcular_grado(matriz):
    grado = np.sum(matriz, axis=0)
    return grado

# Función para graficar conectividad y diagrama de Voronoi con colores según el grado
def graficar_conectividad_voronoi(ax, matriz, canales, puntos_2d, puntos_3d, nombre_archivo):
    # Calcular el grado de cada vértice
    grado = np.sum(matriz, axis=0)

    # Imprimir información del grado en la terminal
    print(f"\nArchivo: {nombre_archivo}")
    for canal, g in zip(canales, grado):
        print(f"Grado de {canal}: {g}")

    # Obtener índices de canales conectados
    conexiones = np.argwhere(matriz == 1)

    # Graficar puntos con colores según el grado
    sc = ax.scatter(puntos_2d[:, 0], puntos_2d[:, 1], c=grado, cmap='Blues')

    # Etiquetar los puntos
    for i in range(len(puntos_2d)):
        ax.text(puntos_2d[i, 0] - 0.02, puntos_2d[i, 1] + 0.025, canales[i])

    # Graficar diagrama de Voronoi
    vor = Voronoi(puntos_2d)
    voronoi_plot_2d(vor, ax=ax, show_points=True, show_vertices=False, s=1)

    # Pintar cada región con un color que represente el grado del vértice
    for r in range(len(vor.point_region)):
        region = vor.regions[vor.point_region[r]]
        if not -1 in region:
            polygon = [vor.vertices[i] for i in region]
            ax.fill(*zip(*polygon), color=plt.cm.Blues(grado[r] / np.max(grado)))

    # Dibujar el círculo que representa la cabeza
    circle = plt.Circle((0, 0), 1.04, color='r', alpha=0.25, fill=False)
    ax.add_patch(circle)

    ax.set_title(f"Voronoi para {nombre_archivo.replace('.txt', '')}")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.axis('equal')

    # Agregar leyenda
    legend = ax.figure.colorbar(sc, ax=ax, orientation='vertical')
    legend.set_label('Número de aristas (Grado)')

#### ARCHIVOS:
# S11 = Emilio Berber
# archivos = ["Lectura_s11.txt", "Memoria_s11.txt", "Operaciones_s11.txt"]
# S09 = Moisés Pineda
# archivos = ["Lectura_s09.txt", "Memoria_s09.txt", "Operaciones_s09.txt"]
# S07 = Samuel B
# archivos = ["Lectura_s07.txt", "Memoria_s07.txt", "Operaciones_s07.txt"]
# S0A = Matriz con 32 electrodos 
# archivos = ["Lectura_s0a.txt", "Memoria_s0a.txt", "Operaciones_s0a.txt"]

# Crear la figura
fig, axs = plt.subplots(1, len(archivos), figsize=(15, 5))

# Mostrar grafos y diagramas de Voronoi para cada archivo .txt
for ax, nombre_archivo in zip(axs, archivos):
    matriz = cargar_matriz(nombre_archivo)

    # Graficar conectividad y diagrama de Voronoi
    graficar_conectividad_voronoi(ax, matriz, channels, points2D, points3D, nombre_archivo)

plt.show()