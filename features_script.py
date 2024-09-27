import numpy as np
from ultralytics import YOLO
import json
from scipy.spatial import ConvexHull
import cv2
from dense.dense import estimate_height_from_point_cloud

# Load a model
model = YOLO("./models/detect-shape_v5.pt")

def calcular_angulo_con_eje_y(normal_plano):
  if len(normal_plano) != 3:
    return -1
  # Vector del eje X
  eje_x = np.array([1, 0, 0])
  
  # Calcular el producto punto
  producto_punto = np.dot(normal_plano, eje_x)
  
  # Calcular las magnitudes de los vectores
  magnitud_normal_plano = np.linalg.norm(normal_plano)
  magnitud_eje_y = np.linalg.norm(eje_x)
  
  # Calcular el coseno del ángulo
  cos_theta = producto_punto / (magnitud_normal_plano * magnitud_eje_y)
  
  # Calcular el ángulo en radianes y luego convertir a grados
  angulo_radianes = np.arccos(cos_theta)
  angulo_grados = np.degrees(angulo_radianes)
  
  return angulo_grados

def get_structure_data(kps, character, confianza, list_tronco_normal, list_head_normal, avg_normal, avg_normal_head, list_centroides, list_union_centroids, centroide, avg_head_centroid, list_is_centroid_to_nariz, list_heights):
    res = {}
    res["persons"] = {}

    for i, person in enumerate(kps):
      list_head = np.array(person)[[0, 1, 2], :].tolist()
      list_tronco = np.array(person)[[5, 6, 11, 12], :].tolist()
      res["persons"][i] = {}
      res["persons"][i]["points"] = person.tolist()
      res["persons"][i]["centroid"] = list_centroides[i].tolist()
      res["persons"][i]["points_tronco"] = list_tronco
      res["persons"][i]["tronco_normal"] = list_tronco_normal[i].tolist()
      res["persons"][i]["angle_tronco"] = calcular_angulo_con_eje_y(list_tronco_normal[i])
      res["persons"][i]["is_centroid_to_nariz"] = list_is_centroid_to_nariz[i]
      res["persons"][i]["points_head"] = list_head
      res["persons"][i]["head_normal"] = list_head_normal[i].tolist()
      res["persons"][i]["angle_head"] = calcular_angulo_con_eje_y(list_head_normal[i])
      res["persons"][i]["height"] = list_heights[i]
    
    res["count"] = i+1
    res["character"] = character
    res["exactitud"] = confianza
    res["centroid"] = centroide.tolist()
    res["avg_normal"] = avg_normal.tolist()
    res["angle_avg_normal"] = calcular_angulo_con_eje_y(avg_normal)
    res["centroid_head"] = avg_head_centroid.tolist()
    res["avg_normal_head"] = avg_normal_head.tolist()
    res["angle_avg_normal_head"] = calcular_angulo_con_eje_y(avg_normal_head)

    new_list_union_centroids = []
    for i in list_union_centroids:
      list_tmp = []
      for j in i:
        list_tmp.append(j.tolist())
      new_list_union_centroids.append(list_tmp)
    res["union_centroids"] = new_list_union_centroids
    
    return json.dumps(res)



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

def get_each_point_of_person(kpts):
    list_points_persons = []
    list_ponits_bodies_nofiltered = []
    # Una por una
    # Ilustrar cada punto en 3D
    for points in kpts:

        # Filtro de puntos menores a 1000 y mayores a 10000
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

        list_ponits_bodies_nofiltered.append(list_body_points)
        list_points_persons.append([filtered_head_points, filtered_body_points])
    return list_points_persons, list_ponits_bodies_nofiltered



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
  # Calculo del producto punto entre el vector y el vector normal del plano
  producto_punto = np.dot(v, normal_plano)
  
  # Si el producto punto es positivo, están en el mismo lado del plano
  return producto_punto > 0

def get_vector_normal_to_head(vector, vector_normal):
    if is_same_side_vector(vector, vector_normal):
        return vector, False
    return get_invest_direction(vector), True

def get_centroid_and_normal(list_points_persons, list_ponits_bodies_nofiltered, list_centroides, list_tronco_normal, list_head_normal, list_is_centroid_to_nariz):
    # Ilustrar los centroides de cada persona
    index=0
    for person in list_points_persons:
        head_points = person[0]
        body_points = person[1]
        # Puede ocasionar que person no pase el filtro por lo que se debe validar
        if len(body_points) < 3:
            list_centroides.append(np.array([]))
            list_tronco_normal.append(np.array([]))
            list_head_normal.append(np.array([]))
            list_is_centroid_to_nariz.append(-1)
            continue
        elif (len(body_points) == 3):
            points_match_body = [(0,1), (0,2), (1,2)]
        else:
            points_match_body = [(0,1), (0,2), (1,3), (2,3)]

        # Calcular centroide del tronco
        centroide = np.mean(np.array(body_points), axis=0)

        # Calcular el vector normal al plano del tronco e ilustrarlo, con el no filtrado para decidir que puntos se usan
        normal = get_vector_normal_to_plane(list_ponits_bodies_nofiltered[index])
        normal = np.array([normal[0], 0, normal[2]])
        if normal is not None:
            list_tronco_normal.append(normal)

            # 0 = nariz, 1 = ojo izquierdo, 2 = ojo derecho
            if head_points[0] and head_points[1] and head_points[2]:
                delta_tmp = (head_points[1][0] - head_points[0][0])
                delta_tmp_2 = (head_points[0][0] - head_points[2][0])
                deltas = [centroide[0], centroide[0] + delta_tmp, centroide[0] - delta_tmp_2]
            elif head_points[0] and head_points[1]:
                delta_tmp = (head_points[1][0] - head_points[0][0])
                deltas = [centroide[0], centroide[0] + delta_tmp]
            elif head_points[0] and head_points[2]:
                delta_tmp = (head_points[0][0] - head_points[2][0])
                deltas = [centroide[0], centroide[0] - delta_tmp]
            elif head_points[1] and head_points[2]:
                delta_tmp = (head_points[1][0] - head_points[2][0]) /2
                deltas = [centroide[0] + delta_tmp, centroide[0] - delta_tmp]
            else:
                deltas = [centroide[0]]
    
            head_points_filtered = [head_pt for head_pt in head_points if head_pt]
            # Si no hay vector normal al plano no se mostrará la nariz y menos el vector normal a la cabeza
            if len(head_points_filtered) > 0:
                individual_head_vector = []
                # calcular el vector de un punto a otro
                for index_head_pt in range(len(head_points_filtered)):
                    head_pt = head_points_filtered[index_head_pt]
                    orientation_tmp = np.array([head_pt[0] - deltas[index_head_pt], head_pt[1] - head_pt[1], head_pt[2] - centroide[2]])
                    orientation_tmp, _is_invest = get_vector_normal_to_head(
                        orientation_tmp, 
                        normal
                    )
                    individual_head_vector.append(orientation_tmp)
                individual_head_avg = np.mean(individual_head_vector, axis=0)
                # individual_head_avg = np.array([individual_head_avg[0], 0, individual_head_avg[2]])
                individual_head_avg, is_invest = get_vector_normal_to_head(
                        individual_head_avg, 
                        normal
                )
                is_centroid_to_nariz = is_invest
                
                list_head_normal.append(individual_head_avg)
                list_is_centroid_to_nariz.append(is_centroid_to_nariz)
            else:
                print("---- No se encuentra la nariz")
                list_head_normal.append(np.array([]))
                list_is_centroid_to_nariz.append(-1)
        else:
            print("---- No hay vector normal al plano")
        list_centroides.append(centroide)

        index+=1


def average_normals(normals):
    new_normals = []
    # Calcular el promedio de los vectores normales
    if len(normals) > 0:
        for i in normals:
            if len(i) == 3:
                new_normals.append(i)
        avg_normal = np.mean(new_normals, axis=0)
        # Normalizar el vector promedio
        avg_normal = avg_normal / np.linalg.norm(avg_normal)
        return avg_normal
    else:
        return None

def get_character(image):
    res = model(image)
    print("Letra detectada: " +  res[0].names[res[0].probs.top1] + " confianza: " + str(res[0].probs.top1conf.item() * 100))
    return res[0].names[res[0].probs.top1], str(res[0].probs.top1conf.item() * 100)

def get_img_shape_meet_prev_sort(list_centroides_sorted, puntos, centroide, list_pos_extremo):
    big_size = 800
    re_size = 640
    half_re_size = re_size//2
    # generar la imagen hasta el limite maximo
    img = np.ones((big_size, big_size, 3), dtype=np.uint8) * 255

    list_points = []
    mean_x = int(centroide[0]) + 400
    mean_y = int(centroide[1])
    list_inf = []

    for simplex in list_centroides_sorted:
        pos0 = simplex[0]
        pos1 = simplex[1]
        p1, p2 = puntos[simplex]
        list_points.append(((int(p1[0])+400, int(p1[1])), (int(p2[0])+400, int(p2[1]))))

        if pos0 in list_pos_extremo[0] or pos1 in list_pos_extremo[0]:
            if pos0 in list_pos_extremo[0] and not pos1 in list_pos_extremo[0]:
                p1 = puntos[pos1]
                p2 = puntos[pos0]
            elif pos1 in list_pos_extremo[0] and not pos0 in list_pos_extremo[0]:
                p1 = puntos[pos0]
                p2 = puntos[pos1]
            list_inf.append(((int(p1[0])+400, int(p1[1])), (int(p2[0])+400, int(p2[1]))))

    # unir los puntos de list_points en la imagen img
    for points in list_points:
        cv2.line(img, points[0], points[1], (0, 0, 0), 2)

    for points in sorted(list_inf, key=lambda x: x[1][0]):
        x1, y1 = points[0]
        x2, y2 = points[1]
        m = (y2 - y1) / (x2 - x1)

        if x1 - x2 != 0 and y1 - y2 != 0:
            arrow_left =  x1 > x2
            if arrow_left:
                x = 0
                b = int(y1 - m * x1)
                cv2.line(img, points[0], [x, b], (0, 0, 0), 2)
                print("arrow_left", x, b)
            else:
                x = big_size-1
                b = int(y2 - m * x2)
                y = int(m * x + b)
                cv2.line(img, points[0], [x, y], (0, 0, 0), 2)
                print("arrow_right", x, y)
        elif y1 - y2 == 0 and x1 - x2 != 0:
            arrow_left =  x1 > x2
            if arrow_left:
                cv2.line(img, points[0], [0, y1], (0, 0, 0), 2)
            else:
                cv2.line(img, points[0], [big_size-1, y2], (0, 0, 0), 2)
        elif x1 - x2 == 0 and y1 - y2 != 0:
            if y1 > y2:
                cv2.line(img, points[0], [x1, 0], (0, 0, 0), 2)
            else:
                cv2.line(img, points[0], [x2, big_size-1], (0, 0, 0), 2)

    if mean_y-half_re_size < 0:
        img_crop = img[:re_size, :]
    elif mean_y+half_re_size > big_size:
        img_crop = img[big_size-re_size:, :]
    else:
        img_crop = img[mean_y-half_re_size:mean_y+half_re_size, :]

    if mean_x-half_re_size < 0:
        img_crop = img_crop[:, :re_size]
    elif mean_x+half_re_size > big_size:
        img_crop = img_crop[:, big_size-re_size:]
    else:
        img_crop = img_crop[:, mean_x-half_re_size:mean_x+half_re_size]

    img_crop = cv2.flip(img_crop, 0)
    img_res = cv2.erode(img_crop, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=15) # _thick
    img_res_2 = cv2.bitwise_not(img_res) # _not_thick

    character, confianza = get_character(img_res_2)
    return character, confianza

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

def get_connection_points(list_centroides, centroide, avg_normal):
    list_centroides = np.array(list_centroides)
    puntos = list_centroides[:,[0,2]]
    centroide_tmp = centroide[[0, 2]]
    avg_normal_tmp = avg_normal[[0, 2]]
    character, confianza = "", 0
    list_pos_extremo = []
    list_union_centroides = []

    if len(puntos) <= 1:
        return [], character, confianza
    elif len(puntos) == 2:
        list_union_centroides = [0, 1]
    else:
        # Eliminar la linea de intersección con el vector normal del grupo
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
    character, confianza = get_img_shape_meet_prev_sort(list_union_centroides, puntos, centroide_tmp, list_pos_extremo)

    # avg_normal
    return list_union_centroides, character, confianza

def get_group_features(list_centroides, centroide, avg_normal, list_head_normal, list_points_persons):
    # Mas de una persona para conectar los puntos
    if len(list_centroides) > 1:
        # Conectar cada uno de los ceintroides y obtiene el 2D de la forma
        print("Show connection points")
        list_union_centroids, character, confianza = get_connection_points(list_centroides, centroide, avg_normal)
    else:
        print("Show connection points: No hay mas de una persona")
        list_union_centroids = []
        character = ""
        confianza = 0

    print("Vector normal promedio")
    # Vector promedio de la cabeza
    if len(list_head_normal) > 0:
        avg_normal_head = average_normals(list_head_normal)
        
        list_nose_height = []
        for i in np.array(list_points_persons, dtype=object)[:, 0]:
            head_points_filtered = [head_pt for head_pt in i if head_pt]
            # A pesar de haber vectores puede que una persona no tenga la nariz detectada, pero list_head_normal sabemos que si tiene al menos una persona completa
            list_nose_height.append(head_points_filtered[0][1])
        
        avg_nose_height = np.mean(list_nose_height)

        # list_points_persons de aqui sacar el promedio de la altura de la nariz
        avg_head_centroid = np.array([centroide[0], avg_nose_height, centroide[2]])

    return avg_normal_head, list_union_centroids, avg_head_centroid, character, confianza

def get_features(keypoints):
    list_heights = []
    list_tronco_normal = []
    list_centroides = []
    list_head_normal = []
    list_is_centroid_to_nariz = []
    centroide = np.array([])
    avg_normal = np.array([0, 0, 0])

    avg_normal_head = np.array([0, 0, 0])
    list_union_centroids = []
    avg_head_centroid = np.array([0, 0, 0])
    character = ""
    confianza = 0

    # Get Height
    for person in keypoints:
        # funcion de altura estimate_height_from_point_cloud
        estimated_height, _centroid = estimate_height_from_point_cloud(point_cloud=person, m_initial=100)
        list_heights.append(estimated_height)

    kps_filtered = np.array(keypoints)[:, [0, 1, 2, 5, 6, 11, 12], :]

    # Get each point of person, all person
    list_points_persons, list_ponits_bodies_nofiltered = get_each_point_of_person(kps_filtered)

    # Get centroid and normal
    get_centroid_and_normal(list_points_persons, list_ponits_bodies_nofiltered, list_centroides, list_tronco_normal, list_head_normal, list_is_centroid_to_nariz)

    if len(list_centroides) > 0:
        # Centroide grupal (centroide del grupo)
        centroide =  np.mean(np.array(list_centroides), axis=0)
        ## Vector promedio del tronco
        avg_normal = average_normals(list_tronco_normal) 
        if avg_normal is not None:
            avg_normal_head, list_union_centroids, avg_head_centroid, character, confianza = get_group_features(list_centroides, centroide, avg_normal, list_head_normal, list_points_persons)

    return get_structure_data(keypoints, character, confianza, list_tronco_normal, list_head_normal, avg_normal, avg_normal_head, list_centroides, list_union_centroids, centroide, avg_head_centroid, list_is_centroid_to_nariz, list_heights)


get_features([np.array([[    -101.09,     -71.567,      277.19],
       [     -98.43,     -76.798,      281.54],
       [       -105,     -75.695,      275.05],
       [     -520.9,     -287.36,      542.82],
       [    -116.56,     -74.086,      275.58],
       [    -105.53,      -44.18,      288.25],
       [    -123.33,     -45.797,      261.37],
       [    -104.27,     -6.9011,      268.34],
       [    -133.79,     -17.095,      250.25],
       [    -96.378,      16.775,      285.34],
       [    -114.01,       10.58,      257.66],
       [    -101.55,      26.453,      272.71],
       [    -114.91,      25.602,      260.48],
       [    -108.19,      81.696,      277.36],
       [    -116.47,      81.067,      262.21],
       [    -113.02,      130.78,       272.5],
       [    -120.18,      135.46,      264.35]]), np.array([[      23.02,     -46.847,      416.17],
       [     26.953,     -50.811,      416.77],
       [     19.635,     -50.943,      417.89],
       [     33.723,     -47.287,      420.22],
       [     15.013,       -47.1,      422.53],
       [     44.793,     -19.462,      413.27],
       [     4.0343,     -19.898,      422.85],
       [     53.653,      14.322,      408.54],
       [     -5.165,      12.532,      420.14],
       [     39.485,      37.564,      407.09],
       [    -2.1087,      37.886,      420.51],
       [     34.006,      41.835,      407.46],
       [     7.3575,      41.821,      411.64],
       [     36.749,      87.827,      414.73],
       [     5.2887,      87.976,      417.58],
       [     37.216,      134.52,      415.15],
       [       10.6,      135.69,      421.68]]), np.array([[    -60.167,      -51.17,      419.82],
       [    -57.238,     -55.656,      420.78],
       [    -64.485,     -55.055,       419.5],
       [    -88.858,     -90.644,      729.41],
       [    -73.328,     -53.279,       422.8],
       [    -44.997,     -27.783,      429.51],
       [    -84.468,     -26.832,      421.32],
       [    -38.154,      7.0596,         433],
       [    -92.802,      7.3138,      421.37],
       [    -44.374,      31.098,      423.37],
       [    -80.758,      29.679,      415.56],
       [    -50.571,       40.45,      419.89],
       [    -75.693,      40.313,       413.8],
       [    -52.647,      90.194,       426.3],
       [    -78.083,      88.766,      415.11],
       [    -54.101,         137,      427.32],
       [    -75.578,      132.71,      410.74]]), np.array([[     60.793,     -61.919,      267.93],
       [     63.351,     -63.616,      260.55],
       [     -520.9,     -287.36,      542.82],
       [     75.881,      -58.81,      254.99],
       [     -520.9,     -287.36,      542.82],
       [     79.311,     -30.753,       235.9],
       [     68.041,     -32.358,      244.72],
       [     78.206,    -0.31798,      230.08],
       [     -520.9,     -287.36,      542.82],
       [     63.478,       26.26,      228.28],
       [     -520.9,     -287.36,      542.82],
       [     73.364,      32.347,       239.7],
       [     61.639,      29.719,      230.67],
       [     71.086,      82.792,      237.07],
       [     65.675,      88.676,      266.96],
       [     81.458,      129.64,      236.31],
       [     72.318,       136.2,       265.3]])])


print([np.array([     9.8223,           0,       7.234]), np.array([   -0.65383,          -0,     -3.1235]), np.array([     3.3021,           0,     -1.0966]), np.array([      8.517,          -0,     -26.491])])

get_features([np.array([[     16.671,     -55.439,      291.66],
       [     19.357,     -59.511,      295.25],
       [     12.981,     -58.834,      291.72],
       [    -390.11,     -215.21,      402.91],
       [      3.761,     -55.646,       292.8],
       [     20.701,     -31.039,      296.74],
       [    -6.6239,     -31.028,      287.51],
       [     19.754,     -1.6286,      291.76],
       [    -18.033,     -4.6857,      290.54],
       [     23.314,      23.152,      294.07],
       [    -8.7028,      19.826,      285.93],
       [     16.855,      27.994,      289.44],
       [   -0.24663,      27.472,      283.77],
       [     13.403,      74.533,      301.68],
       [    -1.7652,      72.317,      287.14],
       [     7.2702,      114.95,       297.6],
       [    -3.0336,      116.57,      292.79]])])