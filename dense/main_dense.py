import os
import cv2
import csv
import joblib
import numpy as np
import dense.pc_generation as pcGen

# Definición de los videos y matrices de configuración
configs = {
    'matlab_1': {
        'LEFT_VIDEO': '../videos/rectified/matlab_1/left_rectified.avi',
        'RIGHT_VIDEO': '../videos/rectified/matlab_1/right_rectified.avi',
        'MATRIX_Q': '../config_files/matlab_1/newStereoMap.xml',
        'disparity_to_depth_map': 'disparity2depth_matrix',
        'model': "../datasets/models/z_estimation_matlab_1_keypoint_ln_model.pkl",
        'numDisparities': 68,
        'blockSize': 7, 
        'minDisparity': 5,
        'disp12MaxDiff': 33,
        'uniquenessRatio': 10,
        'preFilterCap': 33,
        'mode': cv2.StereoSGBM_MODE_HH
    },
    'opencv_1': {
        'LEFT_VIDEO': '../videos/rectified/opencv_1/left_rectified.avi',
        'RIGHT_VIDEO': '../videos/rectified/opencv_1/right_rectified.avi',
        'MATRIX_Q': '../config_files/opencv_1/stereoMap.xml',
        'disparity_to_depth_map': 'disparityToDepthMap',
        'model': "../datasets/models/z_estimation_opencv_1_keypoint_ln_model.pkl",
        'numDisparities': 52,
        'blockSize': 10, 
        'minDisparity': 0,
        'disp12MaxDiff': 36,
        'uniquenessRatio': 39,
        'preFilterCap': 25,
        'mode': cv2.StereoSGBM_MODE_HH
    },
    'matlab_2': {
        'LEFT_VIDEO': '../videos/rectified/matlab_2/left_rectified.avi',
        'RIGHT_VIDEO': '../videos/rectified/matlab_2/right_rectified.avi',
        'MATRIX_Q': '../config_files/laser_config/including_Y_rotation_random/iyrrStereoMap.xml',
        'disparity_to_depth_map': 'disparity2depth_matrix',
        'model': "../datasets/models/z_estimation_matlab_2_keypoint_ln_model.pkl",
        'numDisparities': 68,
        'blockSize': 7, 
        'minDisparity': 5,
        'disp12MaxDiff': 33,
        'uniquenessRatio': 10,
        'preFilterCap': 33,
        'mode': cv2.StereoSGBM_MODE_HH
    }
}

situations = {
    '150_front': 60,
    '150_bodyside_variant': 150,
    '150_500': 3930,
    '200_front': 720,
    '200_shaking_hands_variant': 750,
    '200_400_front': 4080,
    '200_400_sitdown': 6150,
    '200_400_sitdown_side_variant': 6240,
    '250_front': 1020,
    '250_oneback_one_front_variant': 1050,
    '250_side_variant': 1140,
    '250_350': 4200,
    '250_500': 6900,
    '250_600': 4470,
    '250_600_perspective_variant': 4590,
    '300_front': 1290,
    '350_front': 1530,
    '350_side_variant': 1800,
    '400_front': 2010,
    '400_oneside_variant': 2130,
    '400_120cm_h_variant': 5160,
    '450_front': 2310,
    '450_side_variant': 2370,
    '450_600': 4710,
    '500_front': 2700,
    '500_oneside_variant': 2670,
    '550_front': 3000,
    '550_oneside_variant': 2940,
    '600_front': 3240,
    '600_oneside_variant': 3150
}


# Función para seleccionar configuración de cámara
def select_camera_config(camera_type):
    config = configs[camera_type]
    LEFT_VIDEO = config['LEFT_VIDEO']
    RIGHT_VIDEO = config['RIGHT_VIDEO']
    MATRIX_Q = config['MATRIX_Q']
    
    fs = cv2.FileStorage(MATRIX_Q, cv2.FILE_STORAGE_READ)
    Q = fs.getNode(config['disparity_to_depth_map']).mat()
    fs.release()
    
    return LEFT_VIDEO, RIGHT_VIDEO, Q

def extract_frame(video_path, n_frame):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, n_frame)
    retval, frame = cap.read()
    cap.release()
    if not retval:
        raise ValueError(f"No se pudo leer el frame {n_frame}")
    return frame

# Función para extraer un frame específico de los videos izquierdo y derecho
def extract_image_frame(LEFT_VIDEO, RIGHT_VIDEO, n_frame, color=True, save=True):
    image_l = extract_frame(LEFT_VIDEO, n_frame)
    image_r = extract_frame(RIGHT_VIDEO, n_frame)

    if not color:
        image_l = cv2.cvtColor(image_l, cv2.COLOR_BGR2GRAY)
        image_r = cv2.cvtColor(image_r, cv2.COLOR_BGR2GRAY)

    if save:
        cv2.imwrite(f"../images/image_l_{n_frame}.png", image_l)
        cv2.imwrite(f"../images/image_r_{n_frame}.png", image_r)
    
    return image_l, image_r
    

# Función para extraer frames según la situación y configuración de cámara
def extract_situation_frames(camera_type, situation, color=True, save=True):
    if situation in situations:
        n_frame = situations[situation]
        LEFT_VIDEO, RIGHT_VIDEO, Q = select_camera_config(camera_type)
        return extract_image_frame(LEFT_VIDEO, RIGHT_VIDEO, n_frame, color, save), Q
    else:
        raise ValueError("Situación no encontrada en el diccionario.")

# Flujo principal para todas las situaciones
data = []
camera_type = 'matlab_1'
mask_type = 'keypoint'
is_roi = (mask_type == "roi")
situation = "300_front"
model_path = configs[camera_type]['model']
# Cargar el modelo de regresión lineal entrenado
model = joblib.load(model_path)

# for situation in situations:
#     try:
#         print(f"\nProcesando situación: {situation}")
#         (img_l, img_r), Q = extract_situation_frames(camera_type, situation, False, False)
#         img_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2RGB)
#         img_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2RGB)
        
#         disparity = pcGen.compute_disparity(img_l, img_r, configs[camera_type])

#         # # Generar nube de puntos densa sin filtrado adicional
#         dense_point_cloud, dense_colors = pcGen.disparity_to_pointcloud(disparity, Q, img_l)
#         dense_point_cloud = dense_point_cloud.astype(np.float64)

#         # Corrección de nube densa 
#         dense_point_cloud = pcGen.point_cloud_correction(dense_point_cloud, model)

#         base_filename = f"./point_clouds/{camera_type}/{mask_type}_disparity/{camera_type}_{situation}"
#         pcGen.save_dense_point_cloud(dense_point_cloud, dense_colors, base_filename)

#         # # Generar nube de puntos con filtrado y aplicar DBSCAN
#         point_cloud, colors, eps, min_samples = pcGen.generate_filtered_point_cloud(img_l, disparity, Q, camera_type, use_roi=is_roi)

#         # Correción de nube no densa
#         point_cloud = pcGen.point_cloud_correction(point_cloud, model)
#         #centroids = pcGen.process_point_cloud(point_cloud, eps, min_samples, base_filename)
#         original_filename = f"{base_filename}_original.ply"
#         pcGen.save_point_cloud(point_cloud, colors, original_filename)

#         # z_estimations = [centroid[2] for centroid in centroids] if centroids is not None else []
#         # data.append({
#         #     "situation": situation,
#         #     **{f"z_estimation_{i+1}": z for i, z in enumerate(z_estimations)}
#         # })

        
#     except Exception as e:
#         print(f"Error procesando {situation}: {e}")


# Flujo principal
try:
    img_l, img_r = cv2.imread("../images/calibration_results/rectified_new_14_12_47_13_05_2024_IMG_LEFT.jpg"), cv2.imread("../images/calibration_results/rectified_new_14_12_47_13_05_2024_IMG_RIGHT.jpg")
    MATRIX_Q = configs["matlab_1"]['MATRIX_Q']
    fs = cv2.FileStorage(MATRIX_Q, cv2.FILE_STORAGE_READ)
    Q = fs.getNode(configs["matlab_1"]['disparity_to_depth_map']).mat()
    fs.release()

    # (img_l, img_r), Q = extract_situation_frames(camera_type, situation, False, False)
    img_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2RGB)
    img_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2RGB)

    disparity = pcGen.compute_disparity(img_l, img_r, configs[camera_type])

    # Generar nube de puntos densa sin filtrado adicional
    dense_point_cloud, dense_colors = pcGen.disparity_to_pointcloud(disparity, Q, img_l)
    dense_point_cloud = dense_point_cloud.astype(np.float64)
    dense_point_cloud = pcGen.point_cloud_correction(dense_point_cloud, model)
    # guardar nube de puntos densa completa
    base_filename = f"./point_clouds/{camera_type}/{mask_type}_disparity/{camera_type}_{situation}"
    if not os.path.exists(os.path.dirname(base_filename)):
        os.makedirs(os.path.dirname(base_filename))
    pcGen.save_dense_point_cloud(dense_point_cloud, dense_colors, base_filename)

    # Generar nube de puntos con filtrado y aplicar DBSCAN
    point_cloud_list_correction = []
    point_cloud_list, colors_list, eps, min_samples = pcGen.generate_filtered_point_cloud(img_l, disparity, Q, camera_type,  use_roi=is_roi)
    counter = 0
    for pc, cl in zip(point_cloud_list, colors_list):
        point_cloud = pcGen.point_cloud_correction(pc, model)

        # guardar nube de puntos filtrada y corregida
        #pcGen.process_point_cloud(point_cloud, eps, min_samples, base_filename) #This is DBSCAN process
        original_filename = f"{base_filename}_original_{counter}.ply"
        # colors = original_cloud_colors = np.ones_like(point_cloud) * [255, 0, 0]
        pcGen.save_point_cloud(point_cloud, cl, original_filename)
        counter = 1 + counter
        point_cloud_list_correction.append(point_cloud)

    # point_cloud, colors = pcGen.roi_no_dense_pc(img_l,disparity,Q)
    # print(point_cloud, colors)
except Exception as e:
    print(f"Error procesando {situation}: {e}")

################################################ GUARDAR DATASET ###############################################################

if len(data) > 0:
    # Guardar dataset como CSV
    dataset_path = f"../datasets/data/z_estimation_{camera_type}_{mask_type}.csv"

    if not os.path.exists(os.path.dirname(dataset_path)):
        os.makedirs(os.path.dirname(dataset_path))

    max_z_count = max(len(row) - 1 for row in data) # -1 porque situation no es una columna z_estimation

    fieldnames = ["situation"] + [f"z_estimation_{i+1}" for i in range(max_z_count)]

    with open(dataset_path, "w", newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for row in data:
            writer.writerow(row)
    print(f"Dataset guardado en {dataset_path}")
