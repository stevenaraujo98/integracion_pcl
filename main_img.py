import cv2
import numpy as np
import matplotlib.pyplot as plt
from space_3d import get_centroid_and_normal, get_each_point_of_person, get_connection_points
from consts import configs, size_centroide_centroide, size_vector_centroide, size_centroide_head
from dense.dense import load_config, generate_individual_filtered_point_clouds, rectify_images, estimate_height_from_point_cloud
from tests import get_angulo_with_x, get_character, get_structure_data

lista_colores = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
list_colors = [(255, 0, 255), (0, 255, 255), (255, 0, 0), (0, 0, 0), (255, 255, 0),
               (205, 92, 92), (255, 0, 255), (0, 128, 128), (128, 0, 0), (128, 128, 0), (128, 128, 128)]

figure = None


def setup_plot():
    global figure
    if figure is not None:
        return figure

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_ylim(-450, 450)
    ax.set_xlim(-200, 200)
    ax.set_zlim(-300, 800)
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
    ax.view_init(elev=0, azim=270)  # Top view

    # ax.set_ylim(-20, 30)
    # ax.view_init(elev=270, azim=270)  # Front view


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
    kps_filtered = np.array(kpts)[:, [0, 3, 4, 5, 6, 11, 12], :]

    print("Show each point of person, all person")
    get_each_point_of_person(kps_filtered, list_color_to_paint,
                             list_points_persons, list_ponits_bodies_nofiltered, plot_3d, ax)

    print("Show centroid and normal")
    get_centroid_and_normal(list_points_persons, list_ponits_bodies_nofiltered, list_color_to_paint,
                            list_centroides, list_tronco_normal, list_head_normal, list_is_centroid_to_nariz, plot_3d, ax)

    if len(list_centroides) > 0:
        # Ilustrar el centroide de los centroides (centroide del grupo)
        centroide = np.mean(np.array(list_centroides), axis=0)
        plot_3d(centroide[0], centroide[1], centroide[2], ax, "black",
                s=size_centroide_centroide, marker='o', label="Cg")

        # Vector promedio del tronco
        avg_normal = average_normals(list_tronco_normal)

        if avg_normal is not None:
            # Mas de una persona para conectar los puntos
            if len(list_centroides) > 1:
                # Conectar cada uno de los ceintroides y obtiene el 2D de la forma
                print("Show connection points OK")
                list_union_centroids, character, confianza = get_connection_points(
                    list_centroides, name_common, step_frames, centroide, avg_normal, ax)
            else:
                print("Show connection points: No hay mas de una persona")

            print("Vector normal promedio")
            ax.quiver(centroide[0], centroide[1], centroide[2], avg_normal[0], avg_normal[1],
                      avg_normal[2], length=size_vector_centroide, color='black', label='Normal Promedio')

            # Vector promedio de la cabeza
            if len(list_head_normal) > 0:
                avg_normal_head = average_normals(list_head_normal)

                list_nose_height = []
                for i in np.array(list_points_persons, dtype=object)[:, 0]:
                    # A pesar de haber vectores puede que una persona no tenga la nariz detectada, pero list_head_normal sabemos que si tiene al menos una persona completa
                    if len(i[0]) > 0:
                        list_nose_height.append(i[0][1])

                avg_nose_height = int(np.mean(list_nose_height))

                # list_points_persons de aqui sacar el promedio de la altura de la nariz
                plot_3d(centroide[0], avg_nose_height, centroide[2], ax,
                        "black", s=size_centroide_head, marker='o', label="Cgh")
                ax.quiver(centroide[0], avg_nose_height, centroide[2], avg_normal_head[0], avg_normal_head[1],
                          avg_normal_head[2], length=size_vector_centroide, color='black', label='Normal Promedio')
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

    ax.set_xlabel('$x$', fontsize=30, rotation=0, color='purple')
    ax.set_ylabel('$y$', fontsize=30, color='purple')
    ax.set_zlabel('$z$', fontsize=30, rotation=0, color='purple')
    plt.show()
    return list_points_persons, list_tronco_normal, list_head_normal, avg_normal, avg_normal_head, list_centroides, list_union_centroids, centroide, head_centroid, list_is_centroid_to_nariz, character, confianza


camera_type = 'matlab_1'
mask_type = 'keypoint'
is_roi = (mask_type == "roi")
# Usar el método WLS-SGBM, SGBM, ajusta si es RAFT o SELECTIVE según tu configuración
method = 'WLS-SGBM'
method = 'SELECTIVE'
use_max_disparity = False
normalize = True

# name_common = "13_39_44_19_08_2024_IMG"

# path_img_L = "./datasets/190824/ANGULOS_tronco/300/0/" + name_common + "_LEFT.jpg"
# path_img_R = "./datasets/190824/ANGULOS_tronco/300/0/" + name_common + "_RIGHT.jpg"

# name_common = "13_33_10_19_08_2024_IMG"
# name_common = "13_33_16_19_08_2024_IMG"
# name_common = "13_33_23_19_08_2024_IMG"
# name_common = "13_33_28_19_08_2024_IMG"
# ======================================================================================
name_common = "13_33_38_19_08_2024_IMG"
# name_common = "13_33_49_19_08_2024_IMG"
path_img_L = "./datasets/190824/4 PERSONAS/400/I/" + name_common + "_LEFT.jpg"
path_img_R = "./datasets/190824/4 PERSONAS/400/I/" + name_common + "_RIGHT.jpg"

name_common = "14_12_09_23_08_2024_IMG"
path_img_L = "./datasets/190824/4 PERSONAS/300/C/" + name_common + "_LEFT.jpg"
path_img_R = "./datasets/190824/4 PERSONAS/300/C/" + name_common + "_RIGHT.jpg"

# name_common = "11_58_04_19_08_2024_IMG"
# path_img_L = "./datasets/190824/3 PERSONAS/300/L/" + name_common + "_LEFT.jpg"
# path_img_R = "./datasets/190824/3 PERSONAS/300/L/" + name_common + "_RIGHT.jpg"
# ======================================================================================


# name_common = "13_30_51_19_08_2024_IMG"
# path_img_L = "./datasets/190824/4 PERSONAS/400/C/" + name_common + "_LEFT.jpg"
# path_img_R = "./datasets/190824/4 PERSONAS/400/C/" + name_common + "_RIGHT.jpg"

# name_common = "14_24_25_19_08_2024_IMG"
# path_img_L = "./datasets/190824/Profundidades/300/" + name_common + "_LEFT.jpg"
# path_img_R = "./datasets/190824/Profundidades/300/" + name_common + "_RIGHT.jpg"

# name_common = "16_02_53_09_09_2024_IMG"
# path_img_L = "./datasets/190824/ANGULOS_cabeza/400/10/" + name_common + "_LEFT.jpg"
# path_img_R = "./datasets/190824/ANGULOS_cabeza/400/10/" + name_common + "_RIGHT.jpg"

step_frames = 1

try:
    img_l, img_r = cv2.imread(path_img_L), cv2.imread(path_img_R)

    # Calibracion
    img_l, img_r = rectify_images(img_l, img_r, "MATLAB")

    #######################
    # Cargar configuración desde el archivo JSON
    config = load_config("./dense/profiles/profile1.json")

    point_cloud_list, colors_list, keypoints, res_kp_seg = generate_individual_filtered_point_clouds(
        img_l, img_r, config, method, is_roi, use_max_disparity, normalize)
    ##########################
    list_heights = []

    if len(keypoints) > 0 and len(keypoints[0]) > 0:
        img_cop = cv2.cvtColor(img_l.copy(), cv2.COLOR_RGB2BGR)
        for person in res_kp_seg:
            for x, y in person:
                cv2.circle(img_cop, (int(x), int(y)), 2, (0, 0, 255), 2)

        print("Save kp_image", "images/kp/image_" +
              str(name_common) + str(step_frames) + ".jpg")
        # save gray_iget_angulo_with_xmage
        cv2.imwrite("images/kp/image_" + str(name_common) +
                    str(step_frames) + ".jpg", img_cop)

        for person in keypoints:
            estimated_height, centroid = estimate_height_from_point_cloud(
                point_cloud=person, m_initial=100)
            list_heights.append(estimated_height)

        lists_points_3d, list_tronco_normal, list_head_normal, avg_normal, avg_normal_head, list_centroides, list_union_centroids, centroide, head_centroid, list_is_centroid_to_nariz, character, confianza = live_plot_3d(
            keypoints, name_common, step_frames)

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

        # character = ""
        # if len(list_centroides) > 1:
        #     image = cv2.imread("images/shape/gray_image_" + str(name_common) + str(step_frames) + ".jpg")
        #     character, _ = get_character(image)
        # else:
        #     print("No hay mas de una persona")
        print("Se detectó la letra: ", character,
              " con una confianza de: ", confianza)

        get_structure_data(keypoints, character, list_tronco_normal, list_head_normal, avg_normal, avg_normal_head,
                           list_centroides, list_union_centroids, centroide, head_centroid, list_is_centroid_to_nariz, list_heights)


except Exception as e:
    print(f"Error procesando: {e}")
