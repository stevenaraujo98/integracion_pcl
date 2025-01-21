import cv2
import numpy as np
import matplotlib.pyplot as plt
from space_3d import get_centroid_and_normal, get_each_point_of_person, get_connection_points
from consts import configs, size_centroide_centroide, size_vector_centroide, size_centroide_head, size_vector_centroide_head
# from dense.dense import estimate_height_from_point_cloud#, load_config, generate_individual_filtered_point_clouds, rectify_images
from tests import get_angulo_with_x, get_character, get_structure_data
from get_group import get_roi, is_intercept
from dense.keypoint_extraction import get_keypoints, apply_keypoints_mask
import math
from function_to_intel import xy_to_xyz, estimate_height_from_point_cloud
import glob
import json

lista_colores = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
list_colors = [(255,0,255), (0, 255, 255), (255, 0, 0), (0, 0, 0), (255, 255, 0), (205, 92, 92), (255, 0, 255), (0, 128, 128), (128, 0, 0), (128, 128, 0), (128, 128, 128)]

figure = None
def setup_plot():
    global figure
    if figure is not None:
        return figure

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_ylim(-60, 40)
    ax.set_xlim(-300, 300)
    ax.set_zlim(0, 700)

    # ax.set_box_aspect([1, 0.8, 0.8])  # Proporción [x, y, z]
    figure = fig, ax
    return fig, ax

def plot_3d(x, y, z, ax, color, s=20, marker="o", label=None):
    ax.scatter(x, y, z, color=color, marker=marker, s=s)
    if label:
        ax.text(x, y+25, z, label, color=color)

def clean_plot(ax):
    ax.cla()
    ax.set_ylim(-5, 10)
    ax.set_xlim(-50, 50)
    ax.set_zlim(0, 400)
    # Establecer la vista frontal
    ax.view_init(elev=0, azim=270) # Top view

    # ax.set_ylim(-20, 30)
    # ax.view_init(elev=270, azim=270)  # Front view

def average_normals(normals):
    # descartar vectores nulos
    normals_clean = [arr for arr in normals if arr.size > 0]

    new_normals = []
    # Calcular el promedio de los vectores normales
    if len(normals_clean) > 0:
        for i in normals_clean:
            if len(i) == 3:
                new_normals.append(i)
        avg_normal = np.mean(new_normals, axis=0)
        # Normalizar el vector promedio
        avg_normal = avg_normal / np.linalg.norm(avg_normal)
        return avg_normal
    else:
        return []

def live_plot_3d(kpts, name_common, step_frames, is_view=True):
    if is_view:
        fig, ax = setup_plot()
        clean_plot(ax)
    list_points_persons = []
    list_ponits_bodies_nofiltered = []
    list_color_to_paint = []
    list_head_normal = []
    list_is_centroid_to_nariz = []
    list_tronco_normal = []
    list_centroides = []
    list_union_centroids = []
    avg_normal = np.array([])
    avg_normal_head = np.array([])
    centroide = np.array([])
    head_centroid = np.array([0, 0, 0])
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

    print("Show each point of person, all person")
    if is_view:
        get_each_point_of_person(kps_filtered, list_color_to_paint,
                            list_points_persons, list_ponits_bodies_nofiltered, plot_3d, ax)
    else:
        get_each_point_of_person(kps_filtered, list_color_to_paint,
                            list_points_persons, list_ponits_bodies_nofiltered)
        
    print("Show centroid and normal")
    if is_view:
        get_centroid_and_normal(list_points_persons, list_ponits_bodies_nofiltered, list_color_to_paint,
                                list_centroides, list_tronco_normal, list_head_normal, list_is_centroid_to_nariz, plot_3d, ax)
    else:
        get_centroid_and_normal(list_points_persons, list_ponits_bodies_nofiltered, list_color_to_paint,
                                list_centroides, list_tronco_normal, list_head_normal, list_is_centroid_to_nariz)

    if len(list_centroides) > 0 and len(list_centroides[0]) > 0:
        print("---------------------------------- list_centroides", list_centroides)

        # Ilustrar el centroide de los centroides (centroide del grupo)
        centroide = np.mean(np.array(list_centroides), axis=0)
        if is_view:
            plot_3d(centroide[0], centroide[1], centroide[2], ax, "black",
                    s=size_centroide_centroide, marker='o', label="Cg")

        # Vector promedio del tronco
        avg_normal = average_normals(list_tronco_normal)

        if avg_normal is not None:
            # Mas de una persona para conectar los puntos
            if len(list_centroides) > 1:
                # Conectar cada uno de los ceintroides y obtiene el 2D de la forma
                print("Show connection points OK")
                if is_view:
                    list_union_centroids, character, confianza = get_connection_points(
                        list_centroides, name_common, step_frames, centroide, avg_normal, ax)
                else:
                    list_union_centroids, character, confianza = get_connection_points(
                        list_centroides, name_common, step_frames, centroide, avg_normal)
            else:
                print("Show connection points: No hay mas de una persona")

            print("Vector normal promedio")
            if is_view:
                ax.quiver(centroide[0], centroide[1], centroide[2], avg_normal[0], avg_normal[1],
                        avg_normal[2], length=size_vector_centroide, color='black', label='Normal Promedio')

            # Vector promedio de la cabeza
            if len(list_head_normal) > 0 and len(list_head_normal[0]) > 0:
                avg_normal_head = average_normals(list_head_normal)

                list_nose_height = []
                for i in np.array(list_points_persons, dtype=object)[:, 0]:
                    head_points_filtered = [head_pt for head_pt in i if head_pt]
                    # A pesar de haber vectores puede que una persona no tenga la nariz detectada, pero list_head_normal sabemos que si tiene al menos una persona completa
                    if len(head_points_filtered) > 0:
                        list_nose_height.append(head_points_filtered[0][1])

                avg_nose_height = np.mean(list_nose_height)

                # list_points_persons de aqui sacar el promedio de la altura de la nariz
                if is_view:
                    plot_3d(centroide[0], avg_nose_height, centroide[2], ax,
                            "black", s=size_centroide_head, marker='o', label="Cgh")
                    ax.quiver(centroide[0], avg_nose_height, centroide[2], avg_normal_head[0], avg_normal_head[1], 
                            avg_normal_head[2], length=size_vector_centroide_head, color='black', label='Normal Promedio')
                head_centroid = np.array(
                    [centroide[0], avg_nose_height, centroide[2]])

                # ======================================= Con respecto a la head =================================================================
                # # Mas de una persona para conectar los puntos
                # if len(list_centroides) > 1:
                #     # Conectar cada uno de los ceintroides y obtiene el 2D de la forma
                #     print("Show connection points OK")
                #     list_union_centroids, character, confianza = get_connection_points(list_centroides, name_common, step_frames, head_centroid, avg_normal_head, ax)
                # else:
                #     print("Show connection points: No hay mas de una persona")

    if is_view:
        ax.set_xlabel('$x$', fontsize=30, rotation=0, color='purple')
        ax.set_ylabel('$y$', fontsize=30, color='purple')
        ax.set_zlabel('$z$', fontsize=30, rotation=0, color='purple')
        plt.show()
    return list_points_persons, list_tronco_normal, list_head_normal, avg_normal, avg_normal_head, list_centroides, list_union_centroids, centroide, head_centroid, list_is_centroid_to_nariz, character, confianza

class DotDict:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


# Configuración de la cámara
list_centroides_2D = []
list_centroides_process = []
dict_json_res = {}
count_frames = 0 # cuantos frames van
step_frames = 56 # desde cual empezar
# name_common = "25_01_07_16_25_415452"
name_common = "25_01_07_15_47_212161"
name_common = "25_01_07_16_35_331978"
# name_image = "frame_" + str(step_frames) + "_original"
name_image = name_common + "_original"

# path_img = "./datasets/intel/grupos/formas/3 PERSONAS/400/L/" + name_image + ".jpg"
path_img = "./datasets/intel/grupos/grupos/4/" + name_image + ".jpg"
path_img = "./datasets/intel/grupos/formas/4 PERSONAS/400/C/" + name_image + ".jpg"

depth_intrinsics = DotDict(depth_scale=0.0010000000474974513, fx=631.189453125, fy=631.189453125, ppx=647.0123901367188, ppy=362.94287109375)
depth_scale = depth_intrinsics.depth_scale

print("Inicia bucle")
try:
    color_image = cv2.imread(path_img)
    depth_image = cv2.imread(path_img.replace("_original.jpg", "_depth.png"), cv2.IMREAD_UNCHANGED)       

    keypoints = get_keypoints(color_image)
    # result_image = apply_keypoints_mask(depth_image, keypoints) # Obtener la imagen de profundidad con solo los keypoints

    list_heights = []
    point_cloud_list = []

    if len(keypoints) > 0 and len(keypoints[0]) > 0: # que haya puntos y que al menos una persona tenga kp
        color_image_copy = color_image.copy()
        # depth_image_copy = result_image.copy()

        for person in keypoints:
            for x, y in person:
                cv2.circle(color_image_copy, (int(x), int(y)), 2, (255, 0, 0), 2) # BGR

            try:
                kps_body = np.array(person)[:][[5, 6, 11, 12]]
                centroid_kps_body = np.mean(kps_body, axis=0)
                list_centroides_2D.append(centroid_kps_body)
                cv2.circle(color_image_copy, (int(centroid_kps_body[0]), int(centroid_kps_body[1])), 2, (0, 0, 255), 5) # BGR
            except:
                print("No hay datos para centroide individual")


            filtered_array = xy_to_xyz(person, depth_image, depth_scale, depth_intrinsics, to_unit="cm")
            point_cloud_list.append(filtered_array)

            estimated_height, centroid = estimate_height_from_point_cloud(point_cloud=point_cloud_list[-1], m_initial=100, k=0.01)
            print("-------- estimated_height", estimated_height)
            list_heights.append(estimated_height)
        
        print("Save kp_image", "images/kp/image_" + str(name_common) + "_" + name_image + "_" + str(step_frames) + ".jpg")
        # cv2.imwrite("images/kp/image_" + str(name_common) + "_" + name_image + "_" + str(step_frames) + ".jpg", cv2.cvtColor(color_image_copy, cv2.COLOR_BGR2RGB))
        cv2.imwrite("images/kp/image_" + str(name_common) + "_" + name_image + "_" + str(step_frames) + ".jpg", color_image_copy)
        

        print("******************** Cantidad de personas", len(point_cloud_list))
        list_areas = []
        for person in point_cloud_list:
            x_less, y_less, x_more, y_more = get_roi(person)
            list_areas.append([x_less, y_less, x_more, y_more])

        num_intercept = 0
        for i in range(len(list_areas)):
            for j in range(i+1, len(list_areas)):
                if is_intercept(list_areas[i], list_areas[j]):
                    num_intercept += 1
        
        if num_intercept <= len(list_areas)-1:
            print("--------------- No es un grupo")
        else:
            print("+++++++++++++++ Es un grupo")
        print("Cantidad de intercepciones", num_intercept)


        lists_points_3d, list_tronco_normal, list_head_normal, avg_normal, avg_normal_head, list_centroides, list_union_centroids, centroide, head_centroid, list_is_centroid_to_nariz, character, confianza = live_plot_3d(
            point_cloud_list, name_common, step_frames, is_view=True)

        # Test
        print("******************* Angulos de vectores con respecto al tronco *************************")
        for i in list_tronco_normal:
            get_angulo_with_x(i)

        print("******************* Angulo del vector promedio con respecto al tronco *************************")
        get_angulo_with_x(avg_normal)

        print("******************* Angulos de vectores con respecto al head *************************")
        for i in list_head_normal:
            get_angulo_with_x(i)

        print("******************* Angulo del vector promedio con respecto al head *************************")
        get_angulo_with_x(avg_normal_head)

        character = ""
        if len(list_centroides) > 1:
            image = cv2.imread("images/shape/gray_image_" + str(name_common) + str(step_frames) + ".jpg")
            character, _ = get_character(image)
        else:
            print("No hay mas de una persona")
        print("Se detectó la letra: ", character,
            " con una confianza de: ", confianza)

        dict_res = get_structure_data(point_cloud_list, character, list_tronco_normal, list_head_normal, avg_normal, avg_normal_head,
                        list_centroides, list_union_centroids, centroide, head_centroid, list_is_centroid_to_nariz, list_heights)
        
        # print("-------------------------------- keypoints", keypoints)
        print("-------------------------------- list_centroides", list_centroides)
        # print("point_cloud_list", point_cloud_list)
        list_centroides_process.append(list_centroides)
        dict_json_res[str(step_frames)] = dict_res

    print("*"*20, count_frames)
    count_frames += 1   
    step_frames += 1
    name_image = "frame_" + str(step_frames) + "_original"

    print("List of centroides", list_centroides_process)
    print("List of centroides 2D", list_centroides_2D)

except Exception as e:
    print(f"Error procesando: {e}")
    print("List of centroides", list_centroides_process)
    print("List of centroides", list_centroides_2D)
finally:
    print("Finalizado")
    # print(dict_json_res)
    # write json dict_json_res
    # with open("images/jsons/" + str(name_common) + "_" + name_image + "_" + str(step_frames) + ".json", "w") as outfile:
    #     json.dump(dict_json_res, outfile)
