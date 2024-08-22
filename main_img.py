import cv2
import numpy as np
import matplotlib.pyplot as plt
from space_3d import show_centroid_and_normal, show_each_point_of_person, show_connection_points
from consts import configs, size_centroide_centroide, size_vector_centroide, size_centroide_head
from dense.dense import load_config, generate_individual_filtered_point_clouds, rectify_images
from tests import get_angulo_with_x, get_character, get_structure_data

lista_colores = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
list_colors = [(255,0,255), (0, 255, 255), (255, 0, 0), (0, 0, 0), (255, 255, 0), (205, 92, 92), (255, 0, 255), (0, 128, 128), (128, 0, 0), (128, 128, 0), (128, 128, 128)]

figure = None
def setup_plot():
    global figure
    if figure is not None:
        return figure

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_ylim(-250, 250)
    ax.set_xlim(-100, 100)
    ax.set_zlim(0, 500)
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
    avg_normal = 0
    avg_normal_head = 0
    centroide = (0, 0, 0)
    head_centroid = [0, 0, 0]

    # Agregar a una lista de colores para pintar los puntos de cada persona en caso de ser mas de len(lista_colores)
    for i in range(kpts.shape[0]):
        indice_color = i % len(lista_colores)
        list_color_to_paint.append(lista_colores[indice_color])

    """
    # print("Show all points")
    # for points, color in zip(kpts, list_color_to_paint):
    #     for point in points:
    #         if point[0] == 0 and point[1] == 0:
    #             continue
    #         plot_3d(point[0], point[1], point[2], ax, color)
    """

    print("Show each point of person, all person")
    show_each_point_of_person(kpts, list_color_to_paint, ax, plot_3d, list_points_persons, list_ponits_bodies_nofiltered)

    print("Show centroid and normal")
    show_centroid_and_normal(list_points_persons, list_ponits_bodies_nofiltered, list_color_to_paint, ax, list_centroides, list_tronco_normal, list_head_normal, list_is_centroid_to_nariz, plot_3d)

    if len(list_centroides) > 0:
        # Ilustrar el centroide de los centroides (centroide del grupo)
        centroide =  np.mean(np.array(list_centroides), axis=0)
        plot_3d(centroide[0], centroide[1], centroide[2], ax, "black", s=size_centroide_centroide, marker='o', label="Cg")

        # Mas de una persona para conectar los puntos
        if len(list_centroides) > 1:
            # Conectar cada uno de los ceintroides y obtiene el 2D de la forma
            print("Show connection points OK")
            list_union_centroids = show_connection_points(list_centroides, ax, name_common, step_frames, centroide)
        else:
            print("Show connection points: No hay mas de una persona")
    
        ## Vector promedio del tronco
        avg_normal = average_normals(list_tronco_normal)

        if avg_normal is not None:
            print("Vector normal promedio")
            ax.quiver(centroide[0], centroide[1], centroide[2], avg_normal[0], avg_normal[1], avg_normal[2], length=size_vector_centroide, color='black', label='Normal Promedio')
        
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
                plot_3d(centroide[0], avg_nose_height, centroide[2], ax, "black", s=size_centroide_head, marker='o', label="Cgh")
                ax.quiver(centroide[0], avg_nose_height, centroide[2], avg_normal_head[0], avg_normal_head[1], avg_normal_head[2], length=size_vector_centroide, color='black', label='Normal Promedio')
                head_centroid = [centroide[0], avg_nose_height, centroide[2]]

    plt.show()
    return list_points_persons, list_tronco_normal, list_head_normal, avg_normal, avg_normal_head, list_centroides, list_union_centroids, centroide, head_centroid, list_is_centroid_to_nariz

camera_type = 'matlab_1'
mask_type = 'keypoint'
is_roi = (mask_type == "roi")
# Usar el método WLS-SGBM, SGBM, ajusta si es RAFT o SELECTIVE según tu configuración
method = 'SELECTIVE'
use_max_disparity=False
normalize=False

# name_common = "13_44_04_19_08_2024_IMG"

# path_img_L = "./datasets/190824/ANGULOS/300/90/" + name_common + "_LEFT.jpg"
# path_img_R = "./datasets/190824/ANGULOS/300/90/" + name_common + "_RIGHT.jpg"

name_common = "11_57_57_19_08_2024_IMG"

path_img_L = "./datasets/190824/3 PERSONAS/300/L/" + name_common + "_LEFT.jpg"
path_img_R = "./datasets/190824/3 PERSONAS/300/L/" + name_common + "_RIGHT.jpg"

step_frames = 1

try:
    img_l, img_r = cv2.imread(path_img_L), cv2.imread(path_img_R)

    # Calibracion
    img_l, img_r =  rectify_images(img_l, img_r, "MATLAB")

    #######################
    # Cargar configuración desde el archivo JSON
    config = load_config("./dense/profiles/profile1.json")

    point_cloud_list, colors_list, keypoints = generate_individual_filtered_point_clouds(img_l, img_r, config, method, is_roi, use_max_disparity, normalize)
    ##########################

    if len(keypoints) > 0 and len(keypoints[0]) > 0:
        point_cloud_np = np.array(keypoints)[:, [0, 3, 4, 5, 6, 11, 12], :]
        lists_points_3d, list_tronco_normal, list_head_normal, avg_normal, avg_normal_head, list_centroides, list_union_centroids, centroide, head_centroid, list_is_centroid_to_nariz = live_plot_3d(point_cloud_np, name_common, step_frames)

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

        get_structure_data(keypoints, character, list_tronco_normal, list_head_normal, avg_normal, avg_normal_head, list_centroides, list_union_centroids, centroide, head_centroid, list_is_centroid_to_nariz)


except Exception as e:
    print(f"Error procesando: {e}")

