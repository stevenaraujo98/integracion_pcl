import cv2
import numpy as np
from space_3d import get_centroid_and_normal, get_each_point_of_person, get_connection_points
from consts import configs, size_centroide_centroide, size_vector_centroide, size_centroide_head
from dense.dense import load_config, generate_individual_filtered_point_clouds, rectify_images, estimate_height_from_point_cloud
from tests import calcular_angulo_con_eje_y, get_character, get_structure_data
import glob
import json

lista_colores = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
list_colors = [(255,0,255), (0, 255, 255), (255, 0, 0), (0, 0, 0), (255, 255, 0), (205, 92, 92), (255, 0, 255), (0, 128, 128), (128, 0, 0), (128, 128, 0), (128, 128, 128)]

figure = None

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
    kps_filtered = np.array(kpts)[:, [0, 3, 4, 5, 6, 11, 12], :]
 
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
                    # A pesar de haber vectores puede que una persona no tenga la nariz detectada, pero list_head_normal sabemos que si tiene al menos una persona completa
                    if len(i[0]) > 0:
                        list_nose_height.append(i[0][1])
                
                avg_nose_height = int(np.mean(list_nose_height))
                head_centroid = np.array([centroide[0], avg_nose_height, centroide[2]])

    # plt.show()
    return list_points_persons, list_tronco_normal, list_head_normal, avg_normal, avg_normal_head, list_centroides, list_union_centroids, centroide, head_centroid, list_is_centroid_to_nariz, character, confianza

camera_type = 'matlab_1'
mask_type = 'keypoint'
is_roi = (mask_type == "roi")
# Usar el método WLS-SGBM, SGBM, ajusta si es RAFT o SELECTIVE según tu configuración
method = 'SELECTIVE'
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

#########################################################################################FORMAS#########################################################################################
cantidad_personas = "3"
res["formas"][cantidad_personas] = {}
for distancia in distancias:
    res["formas"][cantidad_personas][distancia] = {}
    for forma in formas:
        res["formas"][cantidad_personas][distancia][forma] = []
        path = "datasets/190824/" + cantidad_personas + " PERSONAS/" + distancia + "/" + forma + "/"
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
        path = "datasets/190824/" + cantidad_personas + " PERSONAS/" + distancia + "/" + forma + "/"
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
#########################################################################################FORMAS#########################################################################################

#########################################################################################Orientacion#########################################################################################
res["orientacion"] = {}
angulos = ["0", "10", "20", "30", "40", "50", "60", "70", "80", "90", "100", "110", "120", "130", "140", "150", "160", "170", "180"]
distancias = ["200", "300", "400"]

for distancia in distancias:
    res["orientacion"][distancia] = {}
    for angulo in angulos:
        res["orientacion"][distancia][angulo] = []
        path = "datasets/190824/ANGULOS_tronco/" + distancia + "/" + angulo + "/"
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

#########################################################################################Orientacion cabeza####################################################################################
res["orientacion_cabeza"] = {}
angulos = ["0", "10", "20", "30", "40", "50", "60", "70", "80", "90", "100", "110", "120", "130", "140", "150", "160", "170", "180"]
distancias = ["200", "300", "400"]

for distancia in distancias:
    res["orientacion_cabeza"][distancia] = {}
    for angulo in angulos:
        res["orientacion_cabeza"][distancia][angulo] = []
        path = "datasets/190824/ANGULOS_cabeza/" + distancia + "/" + angulo + "/"
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

                    res["orientacion_cabeza"][distancia][angulo].append({"angulo_tronco": angulo_tronco, "angulo_head": angulo_head})
            except Exception as e:
                print(f"Error procesando: {e}")


#########################################################################################Centroides#########################################################################################
distancias = ["200", "250", "300", "350", "400", "450", "500", "550", "600"]
for distancia in distancias:
    res["centroide"][distancia] = []
    res["height_167"][distancia] = []
    path = "datasets/190824/Profundidades/" + distancia + "/"
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

#########################################################################################Centroide grupal#########################################################################################
distancias = ["200", "250", "300", "350", "400", "450", "500", "550", "600"]
for distancia in distancias:
    res["centroide_grupal"][distancia] = []
    path = "datasets/190824/Profundidad_grupal/" + distancia + "/"
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

print(json.dumps(res))