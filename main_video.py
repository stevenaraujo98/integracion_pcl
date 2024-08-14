import cv2
import numpy as np
import matplotlib.pyplot as plt
from space_3d import show_centroid_and_normal, show_each_point_of_person, show_connection_points
from consts import configs, size_centroide_centroide, size_vector_centroide, size_centroide_head
from dense.dense import load_config, generate_individual_filtered_point_clouds
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
    # Calcular el promedio de los vectores normales
    if len(normals) > 0:
        avg_normal = np.mean(normals, axis=0)
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
    list_tronco_normal = []
    list_centroides = []
    list_union_centroids = []
    avg_normal = 0
    avg_normal_head = 0
    centroide = (0, 0, 0)

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
    show_centroid_and_normal(list_points_persons, list_ponits_bodies_nofiltered, list_color_to_paint, ax, list_centroides, list_tronco_normal, list_head_normal, plot_3d)

    if len(list_centroides) > 0:
        # Ilustrar el centroide de los centroides (centroide del grupo)
        centroide =  np.mean(np.array(list_centroides), axis=0)
        plot_3d(centroide[0], centroide[1], centroide[2], ax, "black", s=size_centroide_centroide, marker='o', label="Cg")

        # Conectar cada uno de los ceintroides y obtiene el 2D de la forma
        print("Show connection points")
        list_union_centroids = show_connection_points(list_centroides, ax, name_common, step_frames, centroide) 
    
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
                    list_nose_height.append(i[0][1])
                
                avg_nose_height = int(np.mean(list_nose_height))

                # list_points_persons de aqui sacar el promedio de la altura de la nariz
                plot_3d(centroide[0], avg_nose_height, centroide[2], ax, "black", s=size_centroide_head, marker='o', label="Cgh")
                ax.quiver(centroide[0], avg_nose_height, centroide[2], avg_normal_head[0], avg_normal_head[1], avg_normal_head[2], length=size_vector_centroide, color='black', label='Normal Promedio')

    plt.show()
    return list_points_persons, list_tronco_normal, list_head_normal, avg_normal, avg_normal_head, list_centroides, list_union_centroids, centroide

camera_type = 'matlab_1'
mask_type = 'keypoint'
is_roi = (mask_type == "roi")
# Usar el método SGBM, ajusta si es RAFT o SELECTIVE según tu configuración
method = 'SELECTIVE'
use_max_disparity=False
normalize=False

name_common = "16_35_42_26_02_2024_VID_"
path_img_L = "./datasets/Calibrado/" + name_common + "LEFT.avi"
path_img_R = "./datasets/Calibrado/" + name_common + "RIGHT.avi"

video_l = cv2.VideoCapture(path_img_L)
video_r = cv2.VideoCapture(path_img_R)

step_frames = 250 # L # 13
# step_frames = 1000 # I # 11
# step_frames = 705 # C 

try:
    while True:
        step_frames += (10*3) #2
        video_l.set(cv2.CAP_PROP_POS_FRAMES, step_frames)
        video_r.set(cv2.CAP_PROP_POS_FRAMES, step_frames)

        ret_l, frame_l = video_l.read()
        ret_r, frame_r = video_r.read()

        if not ret_l or not ret_r:
            break

        img_l = frame_l
        img_r = frame_r

        print("Frame leído", step_frames)

        #######################
        # Cargar configuración desde el archivo JSON
        config = load_config("./dense/profiles/profile1.json")

        point_cloud_list, colors_list, keypoints = generate_individual_filtered_point_clouds(img_l, img_r, config, method, is_roi, use_max_disparity, normalize)
        ##########################

        if len(keypoints) > 0 and len(keypoints[0]) > 0:
            img_cop = cv2.cvtColor(img_l.copy(), cv2.COLOR_RGB2BGR)
            for person in keypoints:
                for x, y, z in person:
                    cv2.circle(img_cop, (int(x), int(y)), 2, (0, 0, 255), 2)
            print("Save kp_image", "images/kp/image_" + str(name_common) + str(step_frames) + ".jpg")
            # save gray_iget_angulo_with_xmage
            cv2.imwrite("images/kp/image_" + str(name_common) + str(step_frames) + ".jpg", img_cop)
            # cv2.imshow("Left opint", img_cop)
            # cv2.imshow("Left", img_l)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

            point_cloud_np = np.array(keypoints)[:, [0, 3, 4, 5, 6, 11, 12], :]
            lists_points_3d, list_tronco_normal, list_head_normal, avg_normal, avg_normal_head, list_centroides, list_union_centroids, centroide = live_plot_3d(point_cloud_np, name_common, step_frames)

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

            # "images/shape/gray_image_" + str(name_common) + str(step_frames) + ".jpg"
            image = cv2.imread("images/shape/gray_image_" + str(name_common) + str(step_frames) + ".jpg")
            character = get_character(image)

            get_structure_data(keypoints, character, list_tronco_normal, list_head_normal, avg_normal, avg_normal_head, list_centroides, list_union_centroids, centroide)

        else:
            print("No se encontraron puntos")
        break

except Exception as e:
    print(f"Error procesando: {e}")

