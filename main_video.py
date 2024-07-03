import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from space_3d import show_centroid_and_normal, calcular_centroide, show_each_point_of_person, show_connection_points
from consts import configs, size_centroide_centroide
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
    ax.set_ylim(-100, 100)
    ax.set_xlim(-500, 500)
    ax.set_zlim(0, 500)

def live_plot_3d(kpts):
    fig, ax = setup_plot()
    clean_plot(ax)
    list_points_persons = []
    list_color_to_paint = []
    list_tronco_normal = []
    list_centroides = []

    # Agregar a una lista de colores para pintar los puntos de cada persona en caso de ser mas de len(lista_colores)
    for i in range(kpts.shape[0]):
        indice_color = i % len(lista_colores)
        list_color_to_paint.append(lista_colores[indice_color])

    print("Show each point of person, all person")
    show_each_point_of_person(kpts, list_color_to_paint, ax, plot_3d, list_points_persons)
    print("Show centroid and normal")
    show_centroid_and_normal(list_points_persons, list_color_to_paint, ax, list_centroides, list_tronco_normal, plot_3d)

    ## Vector promedio


    if len(list_centroides) > 1:
        # Ilustrar el centroide de los centroides (centroide del grupo)
        centroide = calcular_centroide(list_centroides)
        plot_3d(centroide[0], centroide[1], centroide[2], ax, "black", s=size_centroide_centroide, marker='o', label="Cg")

    # Conectar cada uno de los ceintroides
    print("Show connection points")
    show_connection_points(list_centroides, ax)

    # get_img_shape_meet(list_centroides)
    
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

path_img_L = "./datasets/Calibrado/16_35_42_26_02_2024_VID_LEFT.avi"
path_img_R = "./datasets/Calibrado/16_35_42_26_02_2024_VID_RIGHT.avi"

video_l = cv2.VideoCapture(path_img_L)
video_r = cv2.VideoCapture(path_img_R)

step_frames = 260 # 256
video_l.set(cv2.CAP_PROP_POS_FRAMES, step_frames)
video_r.set(cv2.CAP_PROP_POS_FRAMES, step_frames)


try:
    while True:
        ret_l, frame_l = video_l.read()
        ret_r, frame_r = video_r.read()

        if not ret_l or not ret_r:
            break

        img_l = frame_l
        img_r = frame_r

        

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

        for pc, cl in zip(point_cloud_list, colors_list):
            point_cloud = pcGen.point_cloud_correction(pc, model)
            point_cloud_list_correction.append(point_cloud)

        # YOLO
        # keypointsL_filtered = keypointsL[:, [0, 3, 4, 5, 6, 11, 12], :]
        # keypointsR_filtered = keypointsR_sorted[:, [0, 3, 4, 5, 6, 11, 12], :]

        # # OpenPose
        # # keypointsL_filtered = keypointsL[:, [0, 2, 5, 9, 12], :]
        # # keypointsR_filtered = keypointsR_sorted[:, [0, 2, 5, 9, 12], :]

        # img_cop = img_l.copy()
        # for person in keypoints:
        #     for x, y in person:
        #         cv2.circle(img_cop, (int(x), int(y)), 2, (0, 0, 255), 2)
        # cv2.imshow("Left opint", img_cop)
        # cv2.imshow("Left", img_l)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

        point_cloud_np = np.array(point_cloud_list_correction)[:, [0, 3, 4, 5, 6, 11, 12], :]
        lists_points_3d = live_plot_3d(point_cloud_np)
        break

except Exception as e:
    print(f"Error procesando: {e}")

