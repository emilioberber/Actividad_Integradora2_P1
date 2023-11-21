# Con los grafos de conectividad no ponderados, encuentra los árboles de mínima expansión por el 
# método que gustes que incluyan la mayor cantidad de vértices posibles. En muchos casos, no será 
# posible incluir algunos vértices al estar aislados. El método tiene que ser implementado por los 
# miembros del equipo.

import numpy as np
import matplotlib.pyplot as plt
import sys

# Para las matrices de 8 electrodos: 

channels = ['Fz', 'C3', 'Cz', 'C4', 'Pz', 'PO7', 'Oz', 'PO8']

points3D = [[0, 0.71934, 0.694658], [-0.71934, 0, 0.694658], [0, 0, 1], [0.71934, 0, 0.694658], [0, -0.71934, 0.694658], [-0.587427, -0.808524, -0.0348995], [0, -0.999391, -0.0348995], [0.587427, -0.808524, -0.0348995]]


points3D = np.array(points3D)


# Fórmulas para pasar de 3D a 2D
r = np.sqrt(points3D[:, 0]**2 + points3D[:, 1]**2 + points3D[:, 2]**2)
t = r / (r + points3D[:, 2])
x = t * points3D[:, 0]
y = t * points3D[:, 1]
points2D = np.column_stack((x, y))


class Graph():
    def __init__(self, vertices):
        self.V = vertices
        self.graph = np.zeros((vertices, vertices), dtype=int)

    def printMST(self, parent, channels, points_2d, ax, filename):
        print(f"\nPRIM for: {filename}\n-- Arista ------------- Peso -- ")
        total_peso = 0  # Inicializar el total del peso

        for i in range(1, self.V):
            origen = parent[i]
            destino = i
            distancia = calcular_distancia(points3D[origen], points3D[destino])

            total_peso += distancia  # Sumar al total del peso

            # Determinar el color de la arista
            edge_color = 'red' if self.graph[origen][destino] == 1 or self.graph[destino][origen] == 1 else 'black'

            print(f"{channels[origen]} - {channels[destino]}","\t|\t",f"{distancia:.2f}")

            # Dibuja las aristas en el gráfico 2D
            point_origen = points_2d[origen]
            point_destino = points_2d[destino]
            ax.plot([point_origen[0], point_destino[0]], [point_origen[1], point_destino[1]],color=edge_color, alpha=1)
            #ax.text((point_origen[0] + point_destino[0]) / 2, (point_origen[1] + point_destino[1]) / 2,f'{distancia:.2f}', color='blue')

        # Mostrar el total del peso debajo de todas las aristas
        ax.text(0.5, -0.1, f'Total Peso: {total_peso:.2f}', horizontalalignment='center',verticalalignment='center', transform=ax.transAxes, color='red')

        # Imprimir el total del peso en la terminal
        print(f'Total Peso: {total_peso:.2f}')

    def minKey(self, key, mstSet):
        min_val = sys.maxsize
        min_index = -1

        for v in range(self.V):
            if key[v] < min_val and mstSet[v] == False:
                min_val = key[v]
                min_index = v

        return min_index

    def primMST(self, channels, points_2d, points_3d, ax, filename):
        key = [sys.maxsize] * self.V
        parent = [None] * self.V
        key[0] = 0
        mstSet = [False] * self.V
        parent[0] = -1

        for _ in range(self.V):
            u = self.minKey(key, mstSet)
            mstSet[u] = True

            # Verificar si el nodo actual no está conectado a ningún otro nodo
            if key[u] == sys.maxsize:
                print(f"El nodo {channels[u]} no está conectado a ningún otro nodo. Terminando el algoritmo.")
                break

            for v in range(self.V):
                if self.graph[u][v] == 1 and mstSet[v] == False:
                    distance = calcular_distancia(points_3d[u], points_3d[v])
                    if key[v] > distance:
                        key[v] = distance
                        parent[v] = u

        # Si hay nodos desconectados, mostrar un mensaje
        if None in parent:
            print("Hay nodos que no están conectados a ningún otro nodo.")

        self.printMST(parent, channels, points_2d, ax, filename)



def cargar_matriz(nombre_archivo):
    # Carga del archivo usando numpy.loadtxt
    matriz = np.loadtxt(nombre_archivo, skiprows=0, dtype=int)
    return matriz


def calcular_distancia(punto1, punto2):
    # Fórmula de distancia entre dos puntos en 3D
    return np.sqrt((punto1[0] - punto2[0]) ** 2 + (punto1[1] - punto2[1]) ** 2 + (punto1[2] - punto2[2]) ** 2)


def graficar_conectividad(ax, matriz, canales, puntos_2d, puntos_3d):
    # Obtener índices de canales conectados
    conexiones = np.argwhere(matriz == 1)

    # Graficar puntos
    ax.scatter(puntos_2d[:, 0], puntos_2d[:, 1])

    # Dibujar líneas de conexión con pesos (distancias físicas)
    for conexion in conexiones:
        canal_origen = conexion[0]
        canal_destino = conexion[1]
        punto_origen = puntos_2d[canal_origen]
        punto_destino = puntos_2d[canal_destino]

        # Calcular distancia física en 3D
        distancia = calcular_distancia(puntos_3d[canal_origen], puntos_3d[canal_destino])

        # Dibujar línea con etiqueta del peso
        ax.plot([punto_origen[0], punto_destino[0]], [punto_origen[1], punto_destino[1]],'k-', alpha=0.5)
        ax.text((punto_origen[0] + punto_destino[0]) / 2, (punto_origen[1] + punto_destino[1]) / 2,f'{distancia:.2f}', color='blue')

    # Etiquetar los puntos
    for i in range(len(puntos_2d)):
        ax.text(puntos_2d[i, 0] - 0.02, puntos_2d[i, 1] + 0.025, canales[i])


#### ARCHIVOS:
# S11 = Emilio Berber
archivos = ["Lectura_s11.txt", "Memoria_s11.txt", "Operaciones_s11.txt"]
# S09 = Moisés Pineda
# archivos = ["Lectura_s09.txt", "Memoria_s09.txt", "Operaciones_s09.txt"]
# S07 = Samuel B
# archivos = ["Lectura_s07.txt", "Memoria_s07.txt", "Operaciones_s07.txt"]

# Crear la figura
fig, axs = plt.subplots(1, len(archivos), figsize=(15, 5))

# Mostrar grafos para cada archivo .txt
for ax, nombre_archivo in zip(axs, archivos):
    matriz = cargar_matriz(nombre_archivo)

    # Modificación: Agregar aristas aleatorias para simular nodos sin conexión
    for i in range(1, matriz.shape[0]):
        if np.sum(matriz[i, :]) == 0:  # Si el nodo no tiene conexiones, agrega una conexión aleatoria
            j = np.random.choice(np.arange(matriz.shape[1]))
            matriz[i, j] = 1
            matriz[j, i] = 1

    # Crear el objeto de grafo y asignar la matriz de conexión
    g = Graph(len(channels))
    g.graph = matriz

    # Dibujar puntos y distancias
    graficar_conectividad(ax, matriz, channels, points2D, points3D)

    # Dibujar el MST
    g.primMST(channels, points2D, points3D, ax, nombre_archivo)

    circle = plt.Circle((0, 0), 1.04, color='r', alpha=0.25, fill=False)
    ax.add_patch(circle)
    ax.set_title(nombre_archivo.replace(".txt", ""))
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.axis('equal')

plt.show()