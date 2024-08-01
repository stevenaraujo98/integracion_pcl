import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from space_3d import show_centroid_and_normal, calcular_centroide, show_each_point_of_person, show_connection_points
from consts import configs, size_centroide_centroide, size_vector_centroide
import dense.pc_generation as pcGen
import joblib
# from character_meet import get_img_shape_meet

lista_colores = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
list_colors = [(255,0,255), (0, 255, 255), (255, 0, 0), (0, 0, 0), (255, 255, 0), (205, 92, 92), (255, 0, 255), (0, 128, 128), (128, 0, 0), (128, 128, 0), (128, 128, 128)]
# Parametos intrinsecos de la camara
baseline = 58
f_px = 3098.3472392388794
# f_px = (1430.012149825778 + 1430.9237520924735 + 1434.1823778243315 + 1435.2411841383973) / 4 
# center = ((929.8117736131715 + 936.7865692255547) / 2,
#            (506.4104424162777 + 520.0461674300153) / 2)
center = (777.6339950561523,539.533634185791)



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
    list_tronco_normal = []
    list_centroides = []

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
    show_centroid_and_normal(list_points_persons, list_ponits_bodies_nofiltered, list_color_to_paint, ax, list_centroides, list_tronco_normal, plot_3d)

    if len(list_centroides) > 1:
        # Ilustrar el centroide de los centroides (centroide del grupo)
        centroide = calcular_centroide(list_centroides)
        plot_3d(centroide[0], centroide[1], centroide[2], ax, "black", s=size_centroide_centroide, marker='o', label="Cg")

        # Conectar cada uno de los ceintroides y obtiene el 2D de la forma
        print("Show connection points")
        show_connection_points(list_centroides, ax, name_common, step_frames, centroide) 
    
        ## Vector promedio
        avg_normal = average_normals(list_tronco_normal)
        print("Vector normal promedio")

        # Graficar el vector normal promedio en el origen
        if avg_normal is not None:
            ax.quiver(centroide[0], centroide[1], centroide[2], avg_normal[0], avg_normal[1], avg_normal[2], length=size_vector_centroide, color='black', label='Normal Promedio')
    elif len(list_centroides) == 1:
        # no se conectan centroides ya que solo hay una persona
        # no se saca promedio ya que es el mismo de la persona
        ax.quiver(list_centroides[0][0], list_centroides[0][1], list_centroides[0][2], list_tronco_normal[0][0], list_tronco_normal[0][1], list_tronco_normal[0][2], length=size_vector_centroide, color='black', label='Normal Promedio')

    plt.show()
    return list_points_persons

data = []
camera_type = 'matlab_1'
mask_type = 'keypoint'
is_roi = (mask_type == "roi")
situation = "300_front"
model_path = configs[camera_type]['model']
# Cargar el modelo de regresión lineal entrenado
model = joblib.load(model_path)

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

        MATRIX_Q = configs["matlab_1"]['MATRIX_Q']
        fs = cv2.FileStorage(MATRIX_Q, cv2.FILE_STORAGE_READ)
        Q = fs.getNode(configs["matlab_1"]['disparity_to_depth_map']).mat()
        fs.release()


        # (img_l, img_r), Q = extract_situation_frames(camera_type, situation, False, False)
        img_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2RGB)
        img_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2RGB)

        disparity = pcGen.compute_disparity(img_l, img_r, configs[camera_type])


        # Generar nube de puntos con filtrado y aplicar DBSCAN
        point_cloud_list_correction = []
        point_cloud_list, colors_list, eps, min_samples, keypoints = pcGen.generate_filtered_point_cloud(img_l, disparity, Q, camera_type,  use_roi=is_roi)
        print("Point cloud list")

        if len(point_cloud_list) > 0 and len(point_cloud_list[0]) > 0:
            for pc, cl in zip(point_cloud_list, colors_list):
                point_cloud = pcGen.point_cloud_correction(pc, model)
                point_cloud_list_correction.append(point_cloud)

            img_cop = cv2.cvtColor(img_l.copy(), cv2.COLOR_RGB2BGR)
            for person in keypoints:
                for x, y in person:
                    cv2.circle(img_cop, (int(x), int(y)), 2, (0, 0, 255), 2)
            print("Save kp_image", "images/kp/image_" + str(name_common) + str(step_frames) + ".jpg")
            # save gray_image
            cv2.imwrite("images/kp/image_" + str(name_common) + str(step_frames) + ".jpg", img_cop)
            # cv2.imshow("Left opint", img_cop)
            # cv2.imshow("Left", img_l)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

            point_cloud_np = np.array(point_cloud_list_correction)[:, [0, 3, 4, 5, 6, 11, 12], :]
            lists_points_3d = live_plot_3d(point_cloud_np, name_common, step_frames)
        else:
            print("No se encontraron puntos")
        break

except Exception as e:
    print(f"Error procesando: {e}")

