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

# Para las matrcies de 32 electrodos:

#channels = ['Fp1','Fp2', 'AF3', 'AF4', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'T7', 'C3', 'Cz', 'C4', 'T8', 'CP5', 'CP1', 'CP2', 'CP6', 'P7', 'P3', 'Pz', 'P4', 'P8', 'PO3', 'PO4', 'O1', 'Oz', 'O2']

#points3D = [[-0.308829,0.950477,-0.0348995], [0.308829,0.950477,-0.0348995], [-0.406247,0.871199,0.275637], [0.406247,0.871199,0.275637], [-0.808524,0.587427,-0.0348995], [-0.545007,0.673028,0.5], [0,0.71934,0.694658], [0.545007,0.673028,0.5], [0.808524,0.587427,-0.0348995], [-0.887888,0.340828,0.309017], [-0.37471,0.37471,0.848048], [0.37471,0.37471,0.848048], [0.887888,0.340828,0.309017], [-0.999391,0,-0.0348995], [-0.71934,0,0.694658], [0,0,1], [0.71934,0,0.694658], [0.999391,0,-0.0348995], [-0.887888,-0.340828,0.309017], [-0.37471,-0.37471,0.848048], [0.37471,-0.37471, 0.848048], [0.887888,-0.340828,0.309017], [-0.808524,-0.587427,-0.0348995], [-0.545007,-0.673028,0.5], [0,-0.71934,0.694658], [0.545007,-0.673028,0.5], [0.808524,-0.587427,-0.0348995], [-0.406247,-0.871199,0.275637], [0.406247,-0.871199,0.275637], [-0.308829,-0.950477,-0.0348995], [0,-0.999391,-0.0348995], [0.308829,-0.950477,-0.0348995]]

points3D = np.array(points3D)


# Fórmulas para pasar de 3D a 2D
r = np.sqrt(points3D[:, 0]**2 + points3D[:, 1]**2 + points3D[:, 2]**2)
t = r / (r + points3D[:, 2])
x = t * points3D[:, 0]
y = t * points3D[:, 1]
points2D = np.column_stack((x, y))

def dfs(graph, start, visited):
    visited[start] = True
    for i in range(len(graph)):
        if graph[start][i] == 1 and not visited[i]:
            dfs(graph, i, visited)

def find_connected_components(graph):
    visited = [False] * len(graph)
    components = []

    for i in range(len(graph)):
        if not visited[i]:
            # Start a new component
            component = []
            dfs(graph, i, visited, component)
            components.append(component)
    
    return components

def dfs(graph, start, visited, component):
    visited[start] = True
    component.append(start)
    for i in range(len(graph)):
        if graph[start][i] == 1 and not visited[i]:
            dfs(graph, i, visited, component)


class Graph():
    def _init_(self, vertices): 
        self.V = vertices
        self.graph = np.zeros((vertices, vertices), dtype=int)



    def minKey(self, key, mstSet, component):
        min_val = sys.maxsize
        min_index = -1

        for i in range(len(component)):
            v = component[i]
            if key[i] < min_val and not mstSet[i]:
                min_val = key[i]
                min_index = i

        return min_index

    def printMST(self, all_edges, channels, points_2d, ax, filename, color='red'):
        print(f"\nPRIM for: {filename}\n-- Arista ------------- Peso -- ")
        total_weight = 0

        for edge in all_edges:
            origin, destination, weight = edge
            total_weight += weight
            print(f"{channels[origin]} - {channels[destination]}", "\t|\t", f"{weight:.2f}")

            point_origin = points_2d[origin]
            point_destination = points_2d[destination]
            ax.plot([point_origin[0], point_destination[0]], [point_origin[1], point_destination[1]], color=color, alpha=1)

        ax.text(0.5, -0.1, f'Total Weight: {total_weight:.2f}', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, color=color)
        print(f"Total Weight: {total_weight:.2f}")

    def primMST(self, channels, points_2d, points_3d, ax, filename):
        components = find_connected_components(self.graph)

        # Encontramos el árbol más grande
        largest_component = max(components, key=len)
        largest_component_size = len(largest_component)

        for component in components:
            if len(component) == largest_component_size:
                # Creamos un subgrafo para el árbol encontrado
                subgraph = self.graph[np.ix_(component, component)]

                # Prim
                key = [sys.maxsize] * len(component)
                parent = [-1] * len(component)
                key[0] = 0
                mstSet = [False] * len(component)

                for count in range(len(component)):
                    u = self.minKey(key, mstSet, component)
                    mstSet[u] = True

                    for v in range(len(component)):
                        if 0 < subgraph[u][v] < key[v] and not mstSet[v]:
                            key[v] = subgraph[u][v]
                            parent[v] = u

                # Calculamos toda la información para los nodos y aristas de nuestro MST
                all_edges = []
                for i in range(1, len(component)):
                    if parent[i] != -1:
                        origin = component[parent[i]]
                        destination = component[i]
                        weight = calcular_distancia(points_3d[origin], points_3d[destination])
                        all_edges.append((origin, destination, weight))

                # Imprimimos en rojo el MST encontrado más grande
                self.printMST(all_edges, channels, points_2d, ax, filename)

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
# S0A = Matriz con 32 electrodos 
# archivos = ["Lectura_s0a.txt", "Memoria_s0a.txt", "Operaciones_s0a.txt"]


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
    num_vertices = len(channels)

    g = Graph()
    g.graph = matriz
    #print("Matrix for file", nombre_archivo, ":\n", matriz)


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