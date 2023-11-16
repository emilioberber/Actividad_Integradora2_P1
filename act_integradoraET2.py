# En este archivo estamos haciendo el código relacionado a la etapa 2 en donde haremos:
# 1. BFS
# 2. DFS
# 3. UCS
# 4. Floyd 

from collections import defaultdict
import numpy as np
from queue import Queue  # Necesario para la cola en BFS fifo
import matplotlib.pyplot as plt

# Para las matrices de 8 electrodos: 

channels = ['Fz', 'C3', 'Cz', 'C4', 'Pz', 'PO7', 'Oz', 'PO8']

points3D = [[0, 0.71934, 0.694658], [-0.71934, 0, 0.694658], [0, 0, 1], [0.71934, 0, 0.694658], [0, -0.71934, 0.694658], [-0.587427, -0.808524, -0.0348995], [0, -0.999391, -0.0348995], [0.587427, -0.808524, -0.0348995]]

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

class TreeNode:
    def __init__(self, parent, v, c):
        self.parent = parent
        self.v = v
        self.c = c

    def __lt__(self, node):
        return False

    def path(self):
        node = self
        path = []
        while node:
            path.insert(0, node.v)
            node = node.parent
        return path

class Graph:
    def __init__(self):
        self.graph = defaultdict(list)

    def addEdge(self, u, v, cost):
        self.graph[u].append((v, cost))
        self.graph[v].append((u, cost))  # Adding this line to make the graph undirected

    def adjacent_vertices(self, v):
        return self.graph[v]

    def DFSUtil(self, v, visited, destination, current_path=[]):
        visited.add(v)
        current_path.append(v)

        if v == destination:
            print("", ' '.join(current_path))
            print("Destination reached!")
            return True

        for neighbour, cost in self.graph[v]:
            if neighbour not in visited:
                if self.DFSUtil(neighbour, visited, destination, current_path):
                    return True

        current_path.pop()
        return False

    def DFS(self, start, destination=None):
        visited = set()
        current_path = []

        if not self.DFSUtil(start, visited, destination, current_path):
            print("\nDestination not reached!")
            
    def bfs(self, v0, vg):
        frontier = Queue()
        frontier.put(TreeNode(None, v0, 0))
        explored_set = set()

        while not frontier.empty():
            node = frontier.get()

            if node.v == vg:
                return {"Path": node.path()}

            if node.v not in explored_set:
                adjacent_vertices = self.adjacent_vertices(node.v)

                for vertex, cost in adjacent_vertices:
                    if vertex not in explored_set:  # Avoid revisiting already explored vertices
                        frontier.put(TreeNode(node, vertex, cost + node.c))

                explored_set.add(node.v)

        return None



# Create a new graph for each file
graph = Graph()

# Function to load a matrix from a file
def cargar_matriz(nombre_archivo):
    matriz = np.loadtxt(nombre_archivo, skiprows=0, dtype=int)
    return matriz

# Function to calculate the distance between two points in 3D space
def calcular_distancia(punto1, punto2):
    return np.sqrt((punto1[0] - punto2[0])**2 + (punto1[1] - punto2[1])**2 + (punto1[2] - punto2[2])**2)

# Function to plot connectivity on a 2D plane
def graficar_conectividad(ax, matriz, canales, puntos_2d, puntos_3d, path=None):
    conexiones = np.argwhere(matriz == 1)

    ax.scatter(puntos_2d[:, 0], puntos_2d[:, 1], color='blue')  # Color azul para los puntos

    for conexion in conexiones:
        canal_origen = canales[conexion[0]]
        canal_destino = canales[conexion[1]]
        punto_origen = puntos_2d[conexion[0]]
        punto_destino = puntos_2d[conexion[1]]

        distancia = calcular_distancia(puntos_3d[conexion[0]], puntos_3d[conexion[1]])

        # Verifica si la conexión pertenece al camino BFS encontrado
        if path and (canal_origen, canal_destino) in path:
            # Dibujar la línea en rojo si es parte del camino BFS
            ax.plot([punto_origen[0], punto_destino[0]], [punto_origen[1], punto_destino[1]], color='red', alpha=1)
        else:
            # Dibujar la línea en negro si no es parte del camino BFS
            ax.plot([punto_origen[0], punto_destino[0]], [punto_origen[1], punto_destino[1]], 'k-', alpha=0.1)

        ax.text((punto_origen[0] + punto_destino[0]) / 2, (punto_origen[1] + punto_destino[1]) / 2, f'{distancia:.2f}', color='blue')

    for i in range(len(puntos_2d)):
        ax.text(puntos_2d[i, 0] - 0.02, puntos_2d[i, 1] + 0.025, canales[i])





#### ARCHIVOS:
# S11 = Emilio Berber
# archivos = ["Lectura_s11.txt", "Memoria_s11.txt", "Operaciones_s11.txt"]
# S09 = Moisés Pineda
archivos = ["Lectura_s09.txt", "Memoria_s09.txt", "Operaciones_s09.txt"]
# S07 = Samuel B
# archivos = ["Lectura_s07.txt", "Memoria_s07.txt", "Operaciones_s07.txt"]
# S0A = Matriz con 32 electrodos 
# archivos = ["Lectura_s0a.txt", "Memoria_s0a.txt", "Operaciones_s0a.txt"]

# Create the figure
fig, axs = plt.subplots(1, len(archivos), figsize=(15, 5))

for ax, nombre_archivo in zip(axs, archivos):
    matriz = cargar_matriz(nombre_archivo)
    
    # Create a new graph for each file
    graph = Graph()
    
    # Added the missing parts related to points2D and conexiones
    conexiones = np.argwhere(matriz == 1)
    
    graficar_conectividad(ax, matriz, channels, points2D, points3D)
    circle = plt.Circle((0, 0), 1.04, color='r', alpha=0.25, fill=False)
    ax.add_patch(circle)
    ax.set_title(nombre_archivo.replace(".txt", ""))
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.axis('equal')

    # Add edges to the graph using channel names
    for conexion in conexiones:
        canal_origen = channels[conexion[0]]
        canal_destino = channels[conexion[1]]
        punto_origen = points2D[conexion[0]]
        punto_destino = points2D[conexion[1]]

        distancia = calcular_distancia(points3D[conexion[0]], points3D[conexion[1]])

        ax.plot([punto_origen[0], punto_destino[0]], [punto_origen[1], punto_destino[1]], 'k-', alpha=0.5)
        ax.text((punto_origen[0] + punto_destino[0]) / 2, (punto_origen[1] + punto_destino[1]) / 2, f'{distancia:.2f}', color='blue')

        graph.addEdge(canal_origen, canal_destino, distancia)

        # Calculate the cost as the average distance between origin and destination points
        costo = calcular_distancia(points3D[conexion[0]], points3D[conexion[1]])

        # Add the edge to the graph
        graph.addEdge(canal_origen, canal_destino, costo)

    # Perform BFS fot the current graph
    print(f"\nFor: {nombre_archivo}")
    print(f"BFS path: ")
    result_bfs = graph.bfs('Fz', 'PO8')  # Definir el nodo inicial y el nodo objetivo
    if result_bfs:
        path_bfs = [(result_bfs['Path'][i], result_bfs['Path'][i+1]) for i in range(len(result_bfs['Path'])-1)]
        print(' '.join(map(str, result_bfs['Path'])))
        print("Destination reached!")

        # Pass the BFS path to the plotting function
        graficar_conectividad(ax, matriz, channels, points2D, points3D, path=path_bfs)
    else:
        print("Nodes not connected")

    # Perform DFS traversal from 'Fz' to 'PO8' on the current graph
    print(f"DFS path:")
    graph.DFS('Fz', destination='PO8')


plt.show()