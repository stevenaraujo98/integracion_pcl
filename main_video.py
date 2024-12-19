import cv2
import numpy as np
import matplotlib.pyplot as plt
from space_3d import get_centroid_and_normal, get_each_point_of_person, get_connection_points
from consts import configs, size_centroide_centroide, size_vector_centroide, size_centroide_head, size_vector_centroide_head
from dense.dense import load_config, generate_individual_filtered_point_clouds, rectify_images, estimate_height_from_point_cloud
from tests import get_angulo_with_x, get_character, get_structure_data
from get_group import get_roi, is_intercept

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
    kps_filtered = np.array(kpts)[:, [0, 1, 2, 5, 6, 11, 12], :]

    print("Show each point of person, all person")
    get_each_point_of_person(kps_filtered, list_color_to_paint,
                             list_points_persons, list_ponits_bodies_nofiltered, plot_3d, ax)

    print("Show centroid and normal")
    get_centroid_and_normal(list_points_persons, list_ponits_bodies_nofiltered, list_color_to_paint,
                            list_centroides, list_tronco_normal, list_head_normal, list_is_centroid_to_nariz, plot_3d, ax)

    if len(list_centroides) > 0:
        print("---------------------------------- list_centroides", list_centroides)

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
                    head_points_filtered = [head_pt for head_pt in i if head_pt]
                    # A pesar de haber vectores puede que una persona no tenga la nariz detectada, pero list_head_normal sabemos que si tiene al menos una persona completa
                    if len(head_points_filtered) > 0:
                        list_nose_height.append(head_points_filtered[0][1])

                avg_nose_height = np.mean(list_nose_height)

                # list_points_persons de aqui sacar el promedio de la altura de la nariz
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

    ax.set_xlabel('$x$', fontsize=30, rotation=0, color='purple')
    ax.set_ylabel('$y$', fontsize=30, color='purple')
    ax.set_zlabel('$z$', fontsize=30, rotation=0, color='purple')
    plt.show()
    return list_points_persons, list_tronco_normal, list_head_normal, avg_normal, avg_normal_head, list_centroides, list_union_centroids, centroide, head_centroid, list_is_centroid_to_nariz, character, confianza


camera_type = 'matlab_1'
mask_type = 'keypoint'
is_roi = (mask_type == "roi")
# Usar el método WLS-SGBM, SGBM, ajusta si es RAFT o SELECTIVE según tu configuración
method = 'SELECTIVE'
method = 'WLS-SGBM'
use_max_disparity=False
normalize=True

name_common = "16_35_42_26_02_2024_VID_"
path_img_L = "./datasets/Calibrado/" + name_common + "LEFT.avi"
path_img_R = "./datasets/Calibrado/" + name_common + "RIGHT.avi"

#--------------------------------------------------
# name_common = "11_04_35_17_10_2024_VID" # [477, 518, 1050, 1150]
name_common = "10_56_20_17_10_2024_VID" # [481, 517, 584, 781, 823, 910, 985, 1059, 1140, 1217, 1250, 1423, 1506, 1611, 1828, 1856, 1903, 2079, 2138, 2782, 2881, 2945, 2994, 3116, 3161, 3260, 3469, 3777, 3880, 3946]
path_img_L = "./datasets/190824/grupos/" + name_common + "_LEFT.avi"
path_img_R = "./datasets/190824/grupos/" + name_common + "_RIGHT.avi"
#--------------------------------------------------

video_l = cv2.VideoCapture(path_img_L)
video_r = cv2.VideoCapture(path_img_R)

# step_frames = 477 # L # 13
# step_frames = 1000 # I # 11
# step_frames = 705 # C 
#--------------------------------------------------
list_step_frames = [
    517, 518, 519, 520, 521, 
    823, 824, 825, 826, 827, 
    930, 931, 932, 933, 934, 
    1140, 1141, 1142, 1143, 1144, 
    1230, 1231, 1232, 1233, 1234, 
    1280, 1281, 1282, 1283, 1284, 
    1423, 1425, 1426, 1427, 1428, 
    1470, 1471, 1472, 1473, 1474, 
    1506, 1507, 1508, 1509, 1510, 
    1611, 1612, 1613, 1614, 1615, 
    1870, 1871, 1872, 1873, 1874, 
    2090, 2091, 2092, 2093, 2094, 
    2881, 2882, 2883, 2884, 2885, 2886, 2888, 2890, 2902, 2904, 2906, 2908, 2910, 2912, 2914,
    3116, 3117, 3118, 3119, 3120, 3121, 3123, 3125, 3127, 3129, 3131, 3133, 3135, 3137, 3139,
    3260, 3261, 3262, 3263, 3264, 3265, 3267, 3269, 3271, 3273,
    3777, 3778, 3779, 3780, 3781, 3783, 3785, 3787, 3789, 3791,
    3800, 3801, 3802, 3803, 3804, 3805, 3806, 3807, 3808, 3809,
]
# buscar frames donde se vea el grupo

# list_step_frames = [1471] # Para paper
# list_step_frames = [3777]
count_frames = 0
list_centroides_2D = []
list_centroides_process = []
#--------------------------------------------------

print("Inicia bucle")
try:
    while True:
        #--------------------------------------------------
        step_frames = list_step_frames[count_frames]
        #--------------------------------------------------

        # step_frames += (10*3) #2
        video_l.set(cv2.CAP_PROP_POS_FRAMES, step_frames)
        video_r.set(cv2.CAP_PROP_POS_FRAMES, step_frames)

        ret_l, frame_l = video_l.read()
        ret_r, frame_r = video_r.read()

        if not ret_l or not ret_r:
            break

        img_l = frame_l
        img_r = frame_r

        print("Frame leído", step_frames)
        cv2.imwrite("./datasets/190824/grupos/" + str(name_common) + str(step_frames) + "_LEFT_original.jpg", img_l)
        cv2.imwrite("./datasets/190824/grupos/" + str(name_common) + str(step_frames)+ "_RIGHT_original.jpg", img_r)

        # Calibracion
        img_l, img_r = rectify_images(img_l, img_r, "MATLAB")

        cv2.imwrite("./datasets/190824/grupos/" + str(name_common) + str(step_frames) + "_LEFT.jpg", cv2.cvtColor(img_l, cv2.COLOR_RGB2BGR))
        cv2.imwrite("./datasets/190824/grupos/" + str(name_common) + str(step_frames)+ "_RIGHT.jpg", cv2.cvtColor(img_r, cv2.COLOR_RGB2BGR))


        #######################
        # Cargar configuración desde el archivo JSON
        config = load_config("./dense/profiles/profile1.json")

        point_cloud_list, colors_list, keypoints, res_kp_seg = generate_individual_filtered_point_clouds(
            img_l, img_r, config, method, is_roi, use_max_disparity, normalize)
        ##########################
        list_heights = []

        if len(keypoints) > 0 and len(keypoints[0]) > 0:
            img_cop = cv2.cvtColor(img_l.copy(), cv2.COLOR_RGB2BGR)

            #--------------------------------------------------
            kps_body = np.array(res_kp_seg)[:, [5, 6, 11, 12]]
            centroid_kps_body = np.mean(kps_body, axis=1)
            list_centroides_2D.append(centroid_kps_body)
            for x, y in centroid_kps_body:
                cv2.circle(img_cop, (int(x), int(y)), 2, (255, 0, 0), 2)
            
            #--------------------------------------------------

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


            print("******************** Cantidad de personas", len(keypoints))
            list_areas = []
            for person in keypoints:
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

            # get_structure_data(keypoints, character, list_tronco_normal, list_head_normal, avg_normal, avg_normal_head,
            #                 list_centroides, list_union_centroids, centroide, head_centroid, list_is_centroid_to_nariz, list_heights)
            
            #--------------------------------------------------
            # print("-------------------------------- keypoints", keypoints)
            # print("-------------------------------- list_centroides", list_centroides)
            list_centroides_process.append(list_centroides)

        if count_frames == len(list_step_frames)-1:
            break
        print("*"*20, count_frames)
        count_frames += 1   
        #--------------------------------------------------
        #break
    print("List of centroides", list_centroides_process)
    print("List of centroides 2D", list_centroides_2D)

except Exception as e:
    print(f"Error procesando: {e}")
    print("List of centroides", list_centroides_process)
    print("List of centroides", list_centroides_2D)


