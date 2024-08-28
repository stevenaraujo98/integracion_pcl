import numpy as np
# from dijkstra import Graph
import networkx as nx
from  character_meet import get_img_shape_meet_prev_sort
from consts import size_centroide, size_vector, size_vector_head, size_centroide_head
from scipy.spatial import ConvexHull

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

def get_each_point_of_person(kpts, list_color_to_paint, list_points_persons, list_ponits_bodies_nofiltered, plot_3d = None, ax= None):
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

        if plot_3d and ax:
            # unir las dos listas de puntos
            for point in [filtered_head_points[0]] + filtered_body_points:
                if len(point) > 0:
                    plot_3d(point[0], point[1], point[2], ax, color)

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

def get_centroid_and_normal(list_points_persons, list_ponits_bodies_nofiltered, list_color_to_paint, list_centroides, list_tronco_normal, list_head_normal, list_is_centroid_to_nariz, plot_3d = None, ax = None):
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

        if ax:
            # Union de los puntos que conforman el tronco
            for point in points_match_body:
                ax.plot([body_points[point[0]][0], body_points[point[1]][0]], 
                        [body_points[point[0]][1], body_points[point[1]][1]], 
                        [body_points[point[0]][2], body_points[point[1]][2]], color)

        # Calcular centroide del tronco
        centroide = np.mean(np.array(body_points), axis=0)

        if plot_3d and ax:
            # Grafica del centroide de la persona
            plot_3d(centroide[0], centroide[1], centroide[2], ax, color, s=size_centroide, marker='o', label="C"+str(index))

        # Calcular el vector normal al plano del tronco e ilustrarlo, con el no filtrado para decidir que puntos se usan
        normal = get_vector_normal_to_plane(list_ponits_bodies_nofiltered[index])
        if normal is not None:
            if ax:
                # Graficar el vector normal al plano del tronco
                ax.quiver(centroide[0], centroide[1], centroide[2], normal[0], normal[1], normal[2], length=size_vector, color=color, label='Normal Promedio')
            list_tronco_normal.append(normal)
        
            # Si no hay vector normal al plano no se mostrará la nariz y menos el vector normal a la cabeza
            if len(head_points[0]) > 0:
                if plot_3d and ax:
                    # graficar la nariz
                    plot_3d(head_points[0][0], head_points[0][1], head_points[0][2], ax, color, s=size_centroide, marker='o', label="C"+str(index))
                    # Centroide a la nariz
                    plot_3d(centroide[0], head_points[0][1], centroide[2], ax, color, s=size_centroide_head, marker='o')

                # La distancia z de la nariz tiene que ser menor al centroide
                if (head_points[0][2] <= centroide[2]):
                    normal_head, is_invest = get_vector_normal_to_head(
                        np.array([head_points[0][0] - centroide[0], head_points[0][1] - head_points[0][1], head_points[0][2] - centroide[2]]), 
                        normal
                    )
                    if is_invest:
                        if ax:
                            # Graficar el vector desde la nariz al centroide
                            ax.quiver(head_points[0][0], head_points[0][1], head_points[0][2], normal_head[0], normal_head[1], normal_head[2], length=size_vector_head, color=color, label='Head Vector')
                        is_centroid_to_nariz = False
                    else:
                        if ax:
                            # Graficar el vector desde el centroide a la nariz
                            ax.quiver(centroide[0], head_points[0][1], centroide[2], normal_head[0], normal_head[1], normal_head[2], length=size_vector_head, color=color, label='Head Vector')
                        is_centroid_to_nariz = True
                else:
                    normal_head, is_invest = get_vector_normal_to_head(
                        np.array([centroide[0] - head_points[0][0], head_points[0][1] - head_points[0][1], centroide[2] - head_points[0][2]]), 
                        normal
                    )

                    if is_invest:
                        if ax:
                            # Graficar el vector desde el centroide a la nariz
                            ax.quiver(centroide[0], head_points[0][1], centroide[2], normal_head[0], normal_head[1], normal_head[2], length=size_vector_head, color=color, label='Head Vector')
                        is_centroid_to_nariz = True
                    else:
                        if ax:
                            # Graficar el vector desde la nariz al centroide
                            ax.quiver(head_points[0][0], head_points[0][1], head_points[0][2], normal_head[0], normal_head[1], normal_head[2], length=size_vector_head, color=color, label='Head Vector')
                        is_centroid_to_nariz = False
                
                list_head_normal.append(normal_head)
                list_is_centroid_to_nariz.append(is_centroid_to_nariz)
            else:
                print("---- No se encuentra la nariz")
                list_head_normal.append(np.array([]))
                list_is_centroid_to_nariz.append(-1)
        else:
            print("---- No hay vector normal al plano")
        list_centroides.append(centroide)

        index+=1

# Función para calcular la intersección de dos segmentos
def line_intersection(p1, p2, q1, q2):
    # p1, p2 son los puntos del segmento del polígono
    # q1, q2 son los puntos del segmento del vector
    A1 = p2[1] - p1[1]
    B1 = p1[0] - p2[0]
    C1 = A1 * p1[0] + B1 * p1[1]

    A2 = q2[1] - q1[1]
    B2 = q1[0] - q2[0]
    C2 = A2 * q1[0] + B2 * q1[1]

    det = A1 * B2 - A2 * B1
    if det == 0:
        return None  # Las líneas son paralelas

    x = (B2 * C1 - B1 * C2) / det
    y = (A1 * C2 - A2 * C1) / det

    # Verificar si la intersección está dentro de los segmentos
    if (min(p1[0], p2[0]) <= x <= max(p1[0], p2[0]) and
        min(p1[1], p2[1]) <= y <= max(p1[1], p2[1]) and
        min(q1[0], q2[0]) <= x <= max(q1[0], q2[0]) and
        min(q1[1], q2[1]) <= y <= max(q1[1], q2[1])):
        return np.array([x, y])
    return None

def get_connection_points(list_centroides, name_common, step_frames, centroide, avg_normal, ax=None):
    list_centroides = np.array(list_centroides)
    puntos = list_centroides[:,[0,2]]
    centroide_tmp = centroide[[0, 2]]
    avg_normal_tmp = avg_normal[[0, 2]]
    character, confianza = "", 0
    val_dif_mayor_menor = 0
    list_pos_extremo = []
    list_union_centroides = []

    if len(puntos) <= 1:
        return [], character, confianza
    elif len(puntos) == 2:
        list_union_centroides = [0, 1]
        p1_3D, p2_3D = list_centroides[list_union_centroides[0]]

        if ax:
            ax.plot([p1_3D[0], p2_3D[0]],
                                [p1_3D[1], p2_3D[1]],
                                [p1_3D[2], p2_3D[2]], color='orange')
    else:
        puntos_sorted = puntos[np.argsort(puntos[:, 1])]
        pt_menor = puntos_sorted[0, 1]
        pt_mayor = puntos_sorted[-1, 1]
        val_dif_mayor_menor = pt_mayor - pt_menor
        if val_dif_mayor_menor <= 30: 
            for i in range(len(puntos_sorted) - 1):
                index = np.where(puntos == puntos_sorted[i])[0][0]
                index_2 = np.where(puntos == puntos_sorted[i+1])[0][0]
                list_union_centroides.append([index, index_2])
            list_pos_extremo.append([np.where(puntos == puntos_sorted[0])[0][0], np.where(puntos == puntos_sorted[-1])[0][0]])
        else:
            # Calcular la envolvente convexa
            hull = ConvexHull(puntos)

            # Calcular el punto final del vector
            vector_end = centroide_tmp + avg_normal_tmp * 1000  # Escalar para que sea largo

            for simplex in hull.simplices:
                p1, p2 = puntos[simplex]
                interseccion = line_intersection(p1, p2, centroide_tmp, vector_end)
                if interseccion is None:
                    list_union_centroides.append(simplex)
                else:
                    list_pos_extremo.append(simplex)

        for simplex in list_union_centroides:
            p1_3D, p2_3D = list_centroides[simplex]
                
            if ax:
                ax.plot([p1_3D[0], p2_3D[0]],
                        [p1_3D[1], p2_3D[1]],
                        [p1_3D[2], p2_3D[2]], color='orange')

    character, confianza = get_img_shape_meet_prev_sort(list_union_centroides, puntos, name_common, step_frames, centroide, list_pos_extremo)

    # avg_normal
    return list_union_centroides, character, confianza
