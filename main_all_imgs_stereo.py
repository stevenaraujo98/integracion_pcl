import cv2
import numpy as np
from space_3d import get_centroid_and_normal, get_each_point_of_person, get_connection_points
from consts import configs, size_centroide_centroide, size_vector_centroide, size_centroide_head
from dense.dense import load_config, generate_individual_filtered_point_clouds, rectify_images, estimate_height_from_point_cloud
from tests import calcular_angulo_con_eje_y, get_character, get_structure_data
import glob
import json
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MeanShift, estimate_bandwidth
import math

lista_colores = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
list_colors = [(255,0,255), (0, 255, 255), (255, 0, 0), (0, 0, 0), (255, 255, 0), (205, 92, 92), (255, 0, 255), (0, 128, 128), (128, 0, 0), (128, 128, 0), (128, 128, 128)]

figure = None

def average_normals(normals):
    # descartar vectores nulos
    normals = np.array(normals)
    normals = normals[~np.isnan(normals).any(axis=1)]

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

def live_plot_3d(kpts, name_common, step_frames):
    list_points_persons = []
    list_ponits_bodies_nofiltered = []
    list_color_to_paint = []
    list_head_normal = []
    list_is_centroid_to_nariz = []
    list_tronco_normal = []
    list_centroides = []
    list_union_centroids = []
    head_centroid = np.array([0, 0, 0])
    avg_normal = np.array([])
    avg_normal_head = np.array([])
    centroide = np.array([])
    character = ""
    confianza = 0

    # Agregar a una lista de colores para pintar los puntos de cada persona en caso de ser mas de len(lista_colores)
    for i in range(len(kpts)):
        indice_color = i % len(lista_colores)
        list_color_to_paint.append(lista_colores[indice_color])
    
    # print("Show all points")
    # for points, color in zip(kpts, list_color_to_paint):
    #     for point in points:
    #         if point[0] == 0 and point[1] == 0:
    #             continue
    #         plot_3d(point[0], point[1], point[2], ax, color)
    kps_filtered = np.array(kpts)[:, [0, 1, 2, 5, 6, 11, 12], :]
 
    print("Get each point of person, all person")
    get_each_point_of_person(kps_filtered, list_color_to_paint, list_points_persons, list_ponits_bodies_nofiltered)

    print("Get centroid and normal")
    get_centroid_and_normal(list_points_persons, list_ponits_bodies_nofiltered, list_color_to_paint, list_centroides, list_tronco_normal, list_head_normal, list_is_centroid_to_nariz)

    if len(list_centroides) > 0:
        # Ilustrar el centroide de los centroides (centroide del grupo)
        centroide =  np.mean(np.array(list_centroides), axis=0)

        # Conectar cada uno de los ceintroides y obtiene el 2D de la forma
        ## Vector promedio del tronco
        avg_normal = average_normals(list_tronco_normal)

        if avg_normal is not None:
            # Mas de una persona para conectar los puntos
            if len(list_centroides) > 1:
                print("Show connection points")
                list_union_centroids, character, confianza = get_connection_points(list_centroides, name_common, step_frames, centroide, avg_normal)
            else:
                print("Show connection points: No hay mas de una persona")

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
                head_centroid = np.array([centroide[0], avg_nose_height, centroide[2]])

    # plt.show()
    return list_points_persons, list_tronco_normal, list_head_normal, avg_normal, avg_normal_head, list_centroides, list_union_centroids, centroide, head_centroid, list_is_centroid_to_nariz, character, confianza

camera_type = 'matlab_1'
mask_type = 'keypoint'
is_roi = (mask_type == "roi")
# Usar el método WLS-SGBM, SGBM, ajusta si es RAFT o SELECTIVE según tu configuración
method = 'SGBM'
use_max_disparity=False
normalize=True


distancias = ["300", "400"]
formas = ["C", "L", "I"]
step_frames = 1


res = {}
res["formas"] = {}
res["orientacion"] = {}
res["orientacion_cabeza"] = {}
res["centroide"] = {}
res["centroide_grupal"] = {}
res["height_167"] = {}
res["grupos"] = {}

folder_dataset = "190824"
folder_dataset = "estereo"

#########################################################################################FORMAS#########################################################################################

cantidad_personas = "3"
res["formas"][cantidad_personas] = {}
for distancia in distancias:
    res["formas"][cantidad_personas][distancia] = {}
    for forma in formas:
        res["formas"][cantidad_personas][distancia][forma] = []
        path = "datasets/" + folder_dataset + "/formas/" + cantidad_personas + " PERSONAS/" + distancia + "/" + forma + "/"
        list_names = glob.glob(path + "*LEFT.jpg")
        for name in list_names:
            name_common = name.split("/")[-1][:23]

            path_img_L = path + name_common + "_LEFT.jpg"
            path_img_R = path + name_common + "_RIGHT.jpg"

            try:
                img_l, img_r = cv2.imread(path_img_L), cv2.imread(path_img_R)

                # Calibracion
                img_l, img_r =  rectify_images(img_l, img_r, "MATLAB")

                #######################
                # Cargar configuración desde el archivo JSON
                config = load_config("./dense/profiles/profile1.json")

                point_cloud_list, colors_list, keypoints, res_kp_seg = generate_individual_filtered_point_clouds(img_l, img_r, config, method, is_roi, use_max_disparity, normalize)
                ##########################

                if len(keypoints) > 0 and len(keypoints[0]) > 0:
                    lists_points_3d, list_tronco_normal, list_head_normal, avg_normal, avg_normal_head, list_centroides, list_union_centroids, centroide, head_centroid, list_is_centroid_to_nariz, character, confianza = live_plot_3d(keypoints, name_common, step_frames)

                    res["formas"][cantidad_personas][distancia][forma].append({"result": character, "confidence": confianza})
            except Exception as e:
                print(f"Error procesando: {e}")

cantidad_personas = "4"
res["formas"][cantidad_personas] = {}
for distancia in distancias:
    res["formas"][cantidad_personas][distancia] = {}
    for forma in formas:
        res["formas"][cantidad_personas][distancia][forma] = []
        path = "datasets/" + folder_dataset + "/formas/" + cantidad_personas + " PERSONAS/" + distancia + "/" + forma + "/"
        list_names = glob.glob(path + "*LEFT.jpg")
        for name in list_names:
            name_common = name.split("/")[-1][:23]

            path_img_L = path + name_common + "_LEFT.jpg"
            path_img_R = path + name_common + "_RIGHT.jpg"

            try:
                img_l, img_r = cv2.imread(path_img_L), cv2.imread(path_img_R)

                # Calibracion
                img_l, img_r =  rectify_images(img_l, img_r, "MATLAB")

                #######################
                # Cargar configuración desde el archivo JSON
                config = load_config("./dense/profiles/profile1.json")

                point_cloud_list, colors_list, keypoints, res_kp_seg = generate_individual_filtered_point_clouds(img_l, img_r, config, method, is_roi, use_max_disparity, normalize)
                ##########################

                if len(keypoints) > 0 and len(keypoints[0]) > 0:
                    lists_points_3d, list_tronco_normal, list_head_normal, avg_normal, avg_normal_head, list_centroides, list_union_centroids, centroide, head_centroid, list_is_centroid_to_nariz, character, confianza = live_plot_3d(keypoints, name_common, step_frames)

                    res["formas"][cantidad_personas][distancia][forma].append({"result": character, "confidence": confianza})
            except Exception as e:
                print(f"Error procesando: {e}")

#########################################################################################Orientacion#########################################################################################

"""
res["orientacion"] = {}
angulos = ["0", "10", "20", "30", "40", "50", "60", "70", "80", "90", "100", "110", "120", "130", "140", "150", "160", "170", "180"]
distancias = ["200", "300", "400"]
distancias = ["300"]#, "400"]

for distancia in distancias:
    res["orientacion"][distancia] = {}
    for angulo in angulos:
        res["orientacion"][distancia][angulo] = []
        path = "datasets/" + folder_dataset + "/ANGULOS_tronco/" + distancia + "/" + angulo + "/"
        list_names = glob.glob(path + "*LEFT.jpg")
        for name in list_names:
            name_common = name.split("/")[-1][:23]

            path_img_L = path + name_common + "_LEFT.jpg"
            path_img_R = path + name_common + "_RIGHT.jpg"

            try:
                img_l, img_r = cv2.imread(path_img_L), cv2.imread(path_img_R)

                # Calibracion
                img_l, img_r =  rectify_images(img_l, img_r, "MATLAB")

                #######################
                # Cargar configuración desde el archivo JSON
                config = load_config("./dense/profiles/profile1.json")

                point_cloud_list, colors_list, keypoints, res_kp_seg = generate_individual_filtered_point_clouds(img_l, img_r, config, method, is_roi, use_max_disparity, normalize)
                ##########################

                if len(keypoints) > 0 and len(keypoints[0]) > 0:
                    lists_points_3d, list_tronco_normal, list_head_normal, avg_normal, avg_normal_head, list_centroides, list_union_centroids, centroide, head_centroid, list_is_centroid_to_nariz, character, confianza = live_plot_3d(keypoints, name_common, step_frames)

                    for i in list_tronco_normal:
                        angulo_tronco = calcular_angulo_con_eje_y(i)
                    
                    for i in list_head_normal:
                        angulo_head = calcular_angulo_con_eje_y(i)

                    res["orientacion"][distancia][angulo].append({"angulo_tronco": angulo_tronco, "angulo_head": angulo_head})
            except Exception as e:
                print(f"Error procesando: {e}")
"""

#########################################################################################Orientacion cabeza####################################################################################

res["orientacion_cabeza"] = {}
angulos = ["0", "10", "20", "30", "40", "50", "60", "70", "80", "90", "100", "110", "120", "130", "140", "150", "160", "170", "180"]
distancias = ["200", "300", "400"]
distancias = ["300", "300_a", "400", "500"]

for distancia in distancias:
    res["orientacion_cabeza"][distancia] = {}
    for angulo in angulos:
        res["orientacion_cabeza"][distancia][angulo] = []
        path = "datasets/" + folder_dataset + "/ANGULOS_cabeza/" + distancia + "/" + angulo + "/"
        list_names = glob.glob(path + "*LEFT.jpg")
        for name in list_names:
            print(name)
            name_common = name.split("/")[-1][:23]

            path_img_L = path + name_common + "_LEFT.jpg"
            path_img_R = path + name_common + "_RIGHT.jpg"

            try:
                img_l, img_r = cv2.imread(path_img_L), cv2.imread(path_img_R)

                # Calibracion
                img_l, img_r =  rectify_images(img_l, img_r, "MATLAB")

                #######################
                # Cargar configuración desde el archivo JSON
                config = load_config("./dense/profiles/profile1.json")

                point_cloud_list, colors_list, keypoints, res_kp_seg = generate_individual_filtered_point_clouds(img_l, img_r, config, method, is_roi, use_max_disparity, normalize)
                ##########################

                if len(keypoints) > 0 and len(keypoints[0]) > 0:
                    lists_points_3d, list_tronco_normal, list_head_normal, avg_normal, avg_normal_head, list_centroides, list_union_centroids, centroide, head_centroid, list_is_centroid_to_nariz, character, confianza = live_plot_3d(keypoints, name_common, step_frames)

                    for i in list_tronco_normal:
                        angulo_tronco = calcular_angulo_con_eje_y(i)
                    
                    for i in list_head_normal:
                        angulo_head = calcular_angulo_con_eje_y(i)

                    res["orientacion_cabeza"][distancia][angulo].append({"angulo_tronco": angulo_tronco, "angulo_head": angulo_head})
            except Exception as e:
                print(f"Error procesando: {e}")
                res["orientacion_cabeza"][distancia][angulo].append({"angulo_tronco": -1, "angulo_head": -1})

#########################################################################################Centroides#########################################################################################

distancias = ["200", "250", "300", "350", "400", "450", "500", "550", "600"]
for distancia in distancias:
    res["centroide"][distancia] = []
    res["height_167"][distancia] = []
    path = "datasets/" + folder_dataset + "/Profundidades/" + distancia + "/"
    list_names = glob.glob(path + "*LEFT.jpg")
    for name in list_names:
        name_common = name.split("/")[-1][:23]

        path_img_L = path + name_common + "_LEFT.jpg"
        path_img_R = path + name_common + "_RIGHT.jpg"

        try:
            img_l, img_r = cv2.imread(path_img_L), cv2.imread(path_img_R)

            # Calibracion
            img_l, img_r =  rectify_images(img_l, img_r, "MATLAB")

            #######################
            # Cargar configuración desde el archivo JSON
            config = load_config("./dense/profiles/profile1.json")

            point_cloud_list, colors_list, keypoints, res_kp_seg = generate_individual_filtered_point_clouds(img_l, img_r, config, method, is_roi, use_max_disparity, normalize)
            ##########################

            for person in keypoints:
                estimated_height, centroid = estimate_height_from_point_cloud(point_cloud=person, m_initial=100)
                res["height_167"][distancia].append({"respuesta": estimated_height})

            if len(keypoints) > 0 and len(keypoints[0]) > 0:
                lists_points_3d, list_tronco_normal, list_head_normal, avg_normal, avg_normal_head, list_centroides, list_union_centroids, centroide, head_centroid, list_is_centroid_to_nariz, character, confianza = live_plot_3d(keypoints, name_common, step_frames)


                res["centroide"][distancia].append({"respuesta": centroide[-1]})
        except Exception as e:
            print(f"Error procesando: {e}")
            res["centroide"][distancia].append({"respuesta": -1})
            res["height_167"][distancia].append({"respuesta": -1})

#########################################################################################Centroide grupal#########################################################################################

distancias = ["200", "250", "300", "350", "400", "450", "500", "550", "600"]
for distancia in distancias:
    res["centroide_grupal"][distancia] = []
    path = "datasets/" + folder_dataset + "/Profundidad_grupal/" + distancia + "/"
    list_names = glob.glob(path + "*LEFT.jpg")
    for name in list_names:
        name_common = name.split("/")[-1][:23]

        path_img_L = path + name_common + "_LEFT.jpg"
        path_img_R = path + name_common + "_RIGHT.jpg"

        try:
            img_l, img_r = cv2.imread(path_img_L), cv2.imread(path_img_R)

            # Calibracion
            img_l, img_r =  rectify_images(img_l, img_r, "MATLAB")

            #######################
            # Cargar configuración desde el archivo JSON
            config = load_config("./dense/profiles/profile1.json")

            point_cloud_list, colors_list, keypoints, res_kp_seg = generate_individual_filtered_point_clouds(img_l, img_r, config, method, is_roi, use_max_disparity, normalize)
            ##########################

            if len(keypoints) > 0 and len(keypoints[0]) > 0:
                lists_points_3d, list_tronco_normal, list_head_normal, avg_normal, avg_normal_head, list_centroides, list_union_centroids, centroide, head_centroid, list_is_centroid_to_nariz, character, confianza = live_plot_3d(keypoints, name_common, step_frames)


                res["centroide_grupal"][distancia].append({"respuesta": centroide[-1]})
        except Exception as e:
            print(f"Error procesando: {e}")

##################################################################################Deteccion de grupos#############################################################################

# heuristica
# "algoritmo del vecino más cercano" (Nearest Neighbor Algorithm)

def get_distancia(punto1, punto2):
  """Calcula la distancia euclidiana entre dos puntos."""
  return math.sqrt((punto2[0] - punto1[0]) ** 2 + (punto2[1] - punto1[1]) ** 2)

def vecino_mas_cercano(puntos, posicion_inicial=0):
    """Encuentra un camino que visita todos los puntos usando el algoritmo del vecino más cercano."""
    if not puntos:
        return [], 0

    # Comenzar desde el primer punto
    camino = [(puntos[posicion_inicial], 0)]
    puntos_restantes = puntos.copy()
    puntos_restantes.pop(posicion_inicial)
    distancia_total = 0

    while puntos_restantes:
        ultimo_punto = camino[-1][0]
        # Encontrar el punto más cercano al último punto en el camino
        punto_mas_cercano = min(puntos_restantes, key=lambda punto: get_distancia(ultimo_punto, punto))
        dist_tmp = get_distancia(ultimo_punto, punto_mas_cercano)
        distancia_total += dist_tmp
        camino.append((punto_mas_cercano, dist_tmp))
        puntos_restantes.remove(punto_mas_cercano)

    # Opcionalmente, regresar al punto de inicio para cerrar el ciclo
    dist_tmp = get_distancia(camino[-1][0], camino[0][0])
    distancia_total += dist_tmp
    camino.append((camino[0][0], dist_tmp))

    return camino, distancia_total

def get_count_group(list_centroides_2D, list_list_centroides):
    grupos_by_escena = []

    for index, puntos_escena_2d in enumerate(list_centroides_2D):
        if len(puntos_escena_2d) == 0 and len(list_list_centroides[index]) == 0:
            grupos_by_escena.append({
                "meanshift": {
                    "2D": -1, 
                    "3D": -1
                },
                "distancia": {
                    "2D": -1,
                    "3D": -1
                }
            })
            continue

        puntos_escena_3d = list_list_centroides[index][:, [0, 2]]
        grupos_meanshift = []
        for puntos in [puntos_escena_2d, puntos_escena_3d]:
            X = StandardScaler().fit_transform(puntos)

            if len(X) < 2:  # Si hay muy pocos puntos
                grupos_meanshift.append(1)  # Asumimos un solo grupo
                continue

            # Estimar el bandwidth
            bandwidth = estimate_bandwidth(X, quantile=0.5)
            if bandwidth == 0.0:
                bandwidth = np.std(X) / 4  # Usar desviación estándar como alternativa

            # Crear el modelo MeanShift
            mean_shift = MeanShift(bandwidth=bandwidth, bin_seeding=True)

            # Ajustar el modelo a los datos
            mean_shift.fit(X)

            # Obtener las etiquetas de los clusters
            labels = mean_shift.labels_

            # Obtener los centros de los clusters
            cluster_centers = mean_shift.cluster_centers_

            grupos_meanshift.append(len(np.unique(labels)))

        # grupos por distancia
        puntos_sorted_2d = puntos_escena_2d[np.argsort(puntos_escena_2d[:, 0])]
        camino_2d, distancia_total_2d = vecino_mas_cercano(puntos_sorted_2d.tolist())
        puntos_sorted_3d = puntos_escena_3d[np.argsort(puntos_escena_3d[:, 1])]
        camino_3d, distancia_total_3d = vecino_mas_cercano(puntos_sorted_3d.tolist())

        grupo_distancia = []
        for camino in [camino_2d, camino_3d]:
            count_groups = 0
            for index in range(len(camino) - 1):
                p1 = camino[index][0]
                p2 = camino[index + 1][0]
                distancia = camino[index + 1][1]
                if distancia > 115:
                    count_groups += 1
            grupo_distancia.append(count_groups)

        grupos_by_escena.append({
            "meanshift": {
                "2D": grupos_meanshift[0], 
                "3D": grupos_meanshift[1]
            },
            "distancia": {
                "2D": grupo_distancia[0],
                "3D": grupo_distancia[1]
            }
        })
    
    return grupos_by_escena

grupos = ["3", "4"]
for grupo_de in grupos:
    path = "datasets/" + folder_dataset + "/grupos/" + grupo_de + "/"
    list_names = glob.glob(path + "*LEFT.jpg")
    list_centroides_2D = []
    list_centroides_3D = []
    for name in list_names:
        name_common = name.split("/")[-1][:23]

        path_img_L = path + name_common + "_LEFT.jpg"
        path_img_R = path + name_common + "_RIGHT.jpg"

        point_cloud_list = []
        list_centroide_2D_frame = []

        try:
            img_l, img_r = cv2.imread(path_img_L), cv2.imread(path_img_R)

            # Calibracion
            img_l, img_r =  rectify_images(img_l, img_r, "MATLAB")

            #######################
            # Cargar configuración desde el archivo JSON
            config = load_config("./dense/profiles/profile1.json")

            point_cloud_list, colors_list, keypoints, res_kp_seg = generate_individual_filtered_point_clouds(img_l, img_r, config, method, is_roi, use_max_disparity, normalize)

            for person in keypoints:
                torso = person[[5, 6, 11, 12]]
                torso = torso[~np.all(torso == 0, axis=1)] # Eliminar los puntos que son 0
                centroide = np.mean(torso, axis=0)
                list_centroide_2D_frame.append(centroide)
            ##########################

            if len(keypoints) > 0 and len(keypoints[0]) > 0:
                lists_points_3d, list_tronco_normal, list_head_normal, avg_normal, avg_normal_head, list_centroide_3D_frame, list_union_centroids, centroide, head_centroid, list_is_centroid_to_nariz, character, confianza = live_plot_3d(keypoints, name_common, step_frames)

            list_centroides_2D.append(np.array(list_centroide_2D_frame))
            list_centroides_3D.append(np.array(list_centroide_3D_frame))

        except Exception as e:
            print(f"Error procesando: {e}")
            print("No se detectaron keypoints")
            list_centroides_2D.append([])
            list_centroides_3D.append([])

    print("list_centroides_2D", list_centroides_2D)
    print("list_centroides_3D", list_centroides_3D)
    grupos_by_escena = get_count_group(list_centroides_2D, list_centroides_3D)
    res["grupos"][grupo_de] = {"respuesta": grupos_by_escena}

##################################################################################################################################################################################

print(json.dumps(res))