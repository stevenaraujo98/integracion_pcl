import numpy as np
# from dijkstra import Graph
import networkx as nx
from  character_meet import get_img_shape_meet_prev_sort
from consts import size_centroide, size_vector, size_vector_head

# Funcion para filtrar los puntos que se encuentren en un rango de 500 y agregar lista vacia en caso de no cumplir
def conditional_append(a, b, c, centroide):
    if (centroide[2] - 500) < c <= (centroide[2] + 500):
        return [a, b, c]
    else:
        return []

# Funcion para filtrar los puntos que se encuentren en un rango y no se dispare al infinito, en este caso 1000 y 10000. Si no cumple se agrega una lista vacia
def get_points_filtered(points):
    list_res = []
    for a, b, c in zip(*points):
        # if 100 < c <= 1000 and  a != 0 and b != 0:
        if a != 0 and b != 0:
            list_res.append([a, b, c])
        else:
            list_res.append([])
    return list_res


def show_each_point_of_person(kpts, list_color_to_paint, ax, plot_3d, list_points_persons, list_ponits_bodies_nofiltered):
    # Una por una
    # Ilustrar cada punto en 3D
    for points, color in zip(kpts, list_color_to_paint):

        # Filtro de puntos menores a 1000 y mayores a 10000
        # filtered_points = [[a, b, c] for a, b, c in zip(*points) if 1000 < c <= 10000]
        filtered_head_points = get_points_filtered([points[:,0][:3], points[:,1][:3], points[:,2][:3]])

        list_body_points = []
        count_no_zero = 0
        for a, b, c in zip(*[points[:,0][3:], points[:,1][3:], points[:,2][3:]]): 
            # if 100 < c <= 1000 and a != 0 and b != 0:
            if a != 0 and b != 0:
                list_body_points.append([a, b, c])
                count_no_zero += 1
            else:
                list_body_points.append([])

        # En caso de que no pase el filtro el body. (Validar en pasos posteriores)
        if count_no_zero == 0 and count_no_zero <= 1:
            continue

        # Centroide de la persona completa
        centroide = np.mean(np.array(list_body_points), axis=0)

        # Filtrar que se encuentren en el rango de 1m
        # filtered_head_points = [conditional_append(a, b, c, centroide) for a, b, c in filtered_head_points]
        tmp_filtered_head_points = []
        for item in filtered_head_points:
            if len(item) > 0 and (centroide[2] - 100) < item[2] <= (centroide[2] + 100):
                tmp_filtered_head_points.append(list(item))
            else:
                tmp_filtered_head_points.append([])
        filtered_head_points = tmp_filtered_head_points

        # va a haber [] vacias dentro de list_body_points
        filtered_body_points = []
        for item in list_body_points:
            if len(item) > 0 and (centroide[2] - 1000) < item[2] <= (centroide[2] + 1000):
                filtered_body_points.append(item)


        # unir las dos listas de puntos
        # for point in [filtered_head_points[0]] + filtered_body_points:
        #     if len(point) > 0:
        #         plot_3d(point[0], point[1], point[2], ax, color)

        list_ponits_bodies_nofiltered.append(list_body_points)
        list_points_persons.append([filtered_head_points, filtered_body_points])

"""
    Obtener el vector normal al plano e ilustrarlo

    Args:
        person (list): Una lista de listas que contiene las coordenadas x, y, z de cada punto.
        centroide (tuple): Una tupla que contiene las coordenadas x, y, z del centroide.
        ax (Axes3D): El objeto Axes3D de Matplotlib.
        color (str): El color de la línea que representa el vector normal al plano.

    Returns:
        None
"""
def get_vector_normal_to_plane(person):
    # Vector perpendicular al plano
    # Puntos que definen el plano en el espacio tridimensional
    if len(person[0]) == 0:
        p1 = np.array(person[3])
        p2 = np.array(person[2])
        p3 = np.array(person[1])
    elif len(person[1]) == 0:
        p1 = np.array(person[2])
        p2 = np.array(person[0])
        p3 = np.array(person[3])
    elif len(person[2]) == 0:
        p1 = np.array(person[1])
        p2 = np.array(person[3])
        p3 = np.array(person[0])
    else: # len(person[3]) == 0 y se puede usar si todas son diferentes de 0
        p1 = np.array(person[0])
        p2 = np.array(person[1])
        p3 = np.array(person[2])
        
    if len(p1) == 0 or len(p2) == 0 or len(p3) == 0:
        return None
    # Calcular el vector normal al plano
    v1 = p2 - p1
    v2 = p3 - p1
    normal = np.cross(v1, v2)

    # Normalizar el vector normal
    normal = normal / np.linalg.norm(normal)
    
    return normal

def get_invest_direction(vector):
  return -vector

def is_same_side_vector(v, normal_plano):
  # Calculamos el producto punto entre el vector y el vector normal del plano
  producto_punto = np.dot(v, normal_plano)
  
  # Si el producto punto es positivo, están en el mismo lado del plano
  return producto_punto > 0

def get_vector_normal_to_head(vector, vector_normal):
    if is_same_side_vector(vector, vector_normal):
        return vector, False
    return get_invest_direction(vector), True


def show_centroid_and_normal(list_points_persons, list_ponits_bodies_nofiltered, list_color_to_paint, ax, list_centroides, list_tronco_normal, list_head_normal, plot_3d):
    # Ilustrar los centroides de cada persona
    index=0
    for person, color in zip(list_points_persons, list_color_to_paint):
        head_points = person[0]
        body_points = person[1]
        # Puede ocasionar que person no pase el filtro por lo que se debe validar
        if len(body_points) < 3:
            continue
        elif (len(body_points) == 3):
            points_match_body = [(0,1), (0,2), (1,2)]
        else:
            points_match_body = [(0,1), (0,2), (1,3), (2,3)]

        # Union de los puntos que conforman el tronco
        for point in points_match_body:
            ax.plot([body_points[point[0]][0], body_points[point[1]][0]], 
                    [body_points[point[0]][1], body_points[point[1]][1]], 
                    [body_points[point[0]][2], body_points[point[1]][2]], color)

        # Calcular centroide del tronco
        centroide = np.mean(np.array(body_points), axis=0)

        # Grafica del centroide de la persona
        plot_3d(centroide[0], centroide[1], centroide[2], ax, color, s=size_centroide, marker='o', label="C"+str(index))

        # Calcular el vector normal al plano del tronco e ilustrarlo, con el no filtrado para decidir que puntos se usan
        normal = get_vector_normal_to_plane(list_ponits_bodies_nofiltered[index])
        if normal is not None:
            # Graficar el vector normal al plano del tronco
            ax.quiver(centroide[0], centroide[1], centroide[2], normal[0], normal[1], normal[2], length=size_vector, color=color, label='Normal Promedio')
            list_tronco_normal.append(normal)
        
            # Si no hay vector normal al plano no se mostrará la nariz y menos el vector normal a la cabeza
            if len(head_points[0]) > 0:
                # graficar la nariz
                plot_3d(head_points[0][0], head_points[0][1], head_points[0][2], ax, color, s=size_centroide, marker='o', label="C"+str(index))
                # Centroide a la nariz
                plot_3d(centroide[0], head_points[0][1], centroide[2], ax, color, s=size_centroide, marker='o')

                # La distancia z de la nariz tiene que ser menor al centroide
                if (head_points[0][2] <= centroide[2]):
                    # # Centroide a la nariz o la media de las orejas
                    # plot_3d(centroide[0], mean_y, centroide[2], ax, color, s=size_centroide, marker='o', label="C"+str(index))
                    # ax.plot([centroide[0], head_points[0][0]], [mean_y, head_points[0][1]], [centroide[2], head_points[0][2]], color)
                    
                    normal_head, is_invest = get_vector_normal_to_head(
                        np.array([head_points[0][0] - centroide[0], head_points[0][1] - head_points[0][1], head_points[0][2] - centroide[2]]), 
                        normal
                    )
                    
                    if is_invest:
                        # Graficar el vector desde la nariz al centroide
                        ax.quiver(head_points[0][0], head_points[0][1], head_points[0][2], normal_head[0], normal_head[1], normal_head[2], length=size_vector_head, color=color, label='Head Vector')
                    else:
                        # Graficar el vector desde el centroide a la nariz
                        ax.quiver(centroide[0], head_points[0][1], centroide[2], normal_head[0], normal_head[1], normal_head[2], length=size_vector_head, color=color, label='Head Vector')
                else:
                    normal_head, is_invest = get_vector_normal_to_head(
                        np.array([centroide[0] - head_points[0][0], head_points[0][1] - head_points[0][1], centroide[2] - head_points[0][2]]), 
                        normal
                    )

                    if is_invest:
                        # Graficar el vector desde el centroide a la nariz
                        ax.quiver(centroide[0], head_points[0][1], centroide[2], normal_head[0], normal_head[1], normal_head[2], length=size_vector_head, color=color, label='Head Vector')
                    else:
                        # Graficar el vector desde la nariz al centroide
                        ax.quiver(head_points[0][0], head_points[0][1], head_points[0][2], normal_head[0], normal_head[1], normal_head[2], length=size_vector_head, color=color, label='Head Vector')
                
                list_head_normal.append(normal_head)
        else:
            print("---- No se encuentra la nariz")
        list_centroides.append(centroide)

        index+=1

def show_connection_points(list_centroides, ax, name_common, step_frames, centroide):
    # Calcular la distancia entre puntos en un plano 3D
    # g = Graph()
    G = nx.Graph() # G.clear()

    """
    # Agregar vértices
    for i in range(len(list_centroides)):
        g.add_vertex(i)
    """

    # Agregar conexiones y distancias entre los centroides
    for i in range(len(list_centroides)):
        for j in range(i+1, len(list_centroides)):
            point1 = list_centroides[i]
            point2 = list_centroides[j]

            distance = np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2 + (point1[2] - point2[2])**2)
            # g.add_edge(i, j, distance)
            G.add_edge(i, j, weight = distance)

    """
    # la distancia minima de cada punto asia cada punto, pero se puede encontrar una sola linea porque la minima de un punto a otro es la misma
    # res_sorted = []
    for start_vertex in range(len(list_centroides)-1):
        distances = g.dijkstra(start_vertex)
        print("Distancias mínimas desde el vértice", start_vertex, distances)
        # Ordenar el diccionario por los valores de los items
        sorted_dict = dict(sorted(distances.items(), key=lambda item: item[1]))
        # res_sorted.append((start_vertex, list(sorted_dict.items())[1][0]))
        print((start_vertex, list(sorted_dict.items())[1][0]))
        print(nx.dijkstra_path(G,start_vertex, list(sorted_dict.items())[1][0]))
    """

    # Escoger el camino más corto entre cada centroide sin repetir
    res_sorted = []
    tmp_res = {}
    for start_vertex in range(len(list_centroides)-1):
        for end_vertex in range(start_vertex, len(list_centroides)):
            if (start_vertex == end_vertex):
                continue
            # nodos para la distancia minima
            # print(nx.dijkstra_path(G, start_vertex, end_vertex))
            # la distancia minima
            # print(nx.shortest_path_length(G, source=start_vertex, target=end_vertex, weight='weight'))
            tmp_res[(start_vertex, end_vertex)] = nx.shortest_path_length(G, source=start_vertex, target=end_vertex, weight='weight')
        sorted_dict = dict(sorted(tmp_res.items(), key=lambda item: item[1]))
        res_sorted.append(list(sorted_dict.items())[0][0])

        # remuevo el primer valor del diccionario
        del tmp_res[res_sorted[-1]]

    list_centroides_sorted = []
    if (len(res_sorted) > 0): 
        # Unir los centroides con líneas
        for i, j in res_sorted:
            ax.plot([list_centroides[i][0], list_centroides[j][0]],
                    [list_centroides[i][1], list_centroides[j][1]],
                    [list_centroides[i][2], list_centroides[j][2]], color='orange')
            list_centroides_sorted.append([list_centroides[i], list_centroides[j]])
    else:
        list_centroides_sorted = list_centroides
    
    # get_img_shape_meet_prev_sort(list_centroides_sorted, name_common, step_frames, centroide)
    return list_centroides_sorted
