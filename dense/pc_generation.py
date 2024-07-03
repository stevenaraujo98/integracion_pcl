import os
import cv2
import torch
import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN
import dense.keypoint_extraction as kp


# Aplicar el filtro bilateral
sigma = 1.5  # Parámetro de sigma utilizado para el filtrado WLS.
lmbda = 8000.0  # Parámetro lambda usado en el filtrado WLS.


def save_image(path, image, image_name, grayscale=False):
    # Asegúrate de que el directorio existe
    if not os.path.exists(path):
        os.makedirs(path)

    # Listar todos los archivos en el directorio
    files = os.listdir(path)

    # Filtrar los archivos que son imágenes (puedes ajustar los tipos según tus necesidades)
    image_files = [f for f in files if f.startswith(image_name)]

    # Determinar el siguiente número para la nueva imagen
    next_number = len(image_files) + 1

    # Crear el nombre del archivo para la nueva imagen
    new_image_filename = f'{image_name}_{next_number}.png'
    # Ruta completa del archivo
    full_path = os.path.join(path, new_image_filename)

    # Convertir a escala de grises si es necesario
    if grayscale:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Guardar la imagen usando cv2.imwrite
    cv2.imwrite(full_path, image)



# --------------------------------------------------- DENSE POINT CLOUD ----------------------------------------------------------
#MAPA DE DISPARIDAD
def compute_disparity(left_image, right_image, config):
    left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
    right_image = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)

    blockSize_var = config['blockSize']
    P1 = 8 * 3 * (blockSize_var ** 2)  
    P2 = 32 * 3 * (blockSize_var ** 2) 

    stereo = cv2.StereoSGBM_create(
        numDisparities = config['numDisparities'],
        blockSize = blockSize_var, 
        minDisparity=config['blockSize'],
        P1=P1,
        P2=P2,
        disp12MaxDiff=config['disp12MaxDiff'],
        uniquenessRatio=config['uniquenessRatio'],
        preFilterCap=config['preFilterCap'],
        mode=config['mode']
    )

    # Calcular el mapa de disparidad de la imagen izquierda a la derecha
    left_disp = stereo.compute(left_image, right_image)
    #.astype(np.float32) / 16.0

    # Crear el matcher derecho basado en el matcher izquierdo para consistencia
    right_matcher = cv2.ximgproc.createRightMatcher(stereo)

    # Calcular el mapa de disparidad de la imagen derecha a la izquierda
    right_disp = right_matcher.compute(right_image, left_image)

    # Crear el filtro WLS y configurarlo
    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=stereo)
    wls_filter.setLambda(lmbda)
    wls_filter.setSigmaColor(sigma)

    # Filtrar el mapa de disparidad utilizando el filtro WLS
    filtered_disp = wls_filter.filter(left_disp, left_image, disparity_map_right=right_disp)

    # Normalización para la visualización o procesamiento posterior
    #filtered_disp = cv2.normalize(src=filtered_disp, dst=filtered_disp, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
    # filtered_disp = np.uint8(filtered_disp)
    return filtered_disp



# REPROYECCIÒN DE DISPARIDAD A 3D

def disparity_to_pointcloud(disparity, Q, image, custom_mask=None):
    points_3D = cv2.reprojectImageTo3D(disparity, Q) 
    mask = disparity > 0

    out_points = []
    out_colors = []
    # if custom_mask is not None:
    #     mask  = custom_mask > 0

    # out_points = points_3D[mask]
    # out_colors = image[mask]

    for x, y in custom_mask:
        x = int(x)
        y = int(y)
        if x > 0 and y > 0:
            out_points.append(points_3D[y - 1, x - 1])
            out_colors.append(image[y - 1, x - 1])
        else:
            out_points.append([0, 0, 0])
            out_colors.append([0, 0, 0])


    # return np.array(out_points), np.array(out_colors)
    return out_points, out_colors

# CREACION DE CENTROIDES
def apply_dbscan(point_cloud, eps, min_samples):
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(point_cloud)
    labels = db.labels_
    return labels

def get_centroids(point_cloud, labels):
    unique_labels = set(labels)
    if -1 in unique_labels:
        unique_labels.remove(-1)
    if not unique_labels:
        print("No hay clusters.")
        return None
    else:
        centroids = []
        for label in unique_labels:
            cluster_points = point_cloud[labels == label]
            centroid = np.mean(cluster_points, axis=0)
            centroids.append(centroid)
        return np.array(centroids)

# CREACIÓN DE NUBE DE PUNTOS DENSA
def create_point_cloud(points, colors=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)
    return pcd

def save_point_cloud(point_cloud, colors, filename):
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    pcd = create_point_cloud(point_cloud, colors)
    o3d.io.write_point_cloud(filename, pcd, print_progress=True)

def process_point_cloud(point_cloud, eps, min_samples, base_filename):
    labels = apply_dbscan(point_cloud, eps, min_samples)
    centroids = get_centroids(point_cloud, labels)

    if centroids is not None:
        
        centroid_colors = np.tile([[255, 0, 0]], (len(centroids), 1))  # Rojo
        centroid_filename = f"{base_filename}_centroids.ply"
        save_point_cloud(centroids, centroid_colors, centroid_filename)
        
    original_cloud_colors = np.ones_like(point_cloud) * [0, 0, 255]  # Azul
    original_filename = f"{base_filename}_original.ply"
    save_point_cloud(point_cloud, original_cloud_colors, original_filename)

    return centroids

def generate_all_filtered_point_cloud(img_l, disparity, Q, camera_type, use_roi=True, ):
    
    if use_roi:
        seg = kp.get_segmentation(img_l)
        result_image = kp.apply_seg_mask(disparity, seg)
        #save_image("../images/prediction_results/", result_image, "filtered_seg", False)
        eps, min_samples = 2, 3500
    else:
        keypoints = kp.get_keypoints(img_l)
        result_image = kp.apply_keypoints_mask(disparity, keypoints)
        #save_image("../images/prediction_results/", result_image, "filtered_keypoints", False)

        eps = 50 if "matlab" in camera_type else 10
        min_samples = 6

    point_cloud, colors = disparity_to_pointcloud(disparity, Q, img_l, result_image)
    point_cloud = point_cloud.astype(np.float64)
    
    return point_cloud, colors, eps, min_samples

def generate_filtered_point_cloud(img_l, disparity, Q, camera_type, use_roi=True, ):
    result_image_list = []
    point_cloud_list = []
    colors_list = []
    if use_roi:
        seg = kp.get_segmentation(img_l)
        for i in seg:
            i_list = [i]
            result_image = kp.apply_seg_mask(disparity, i_list)
            result_image_list.append(result_image)
            
        #save_image("../images/prediction_results/", result_image, "filtered_seg", False)
        eps, min_samples = 2, 3500
    else:
        keypoints = kp.get_keypoints(img_l)
        # for i in keypoints:
        #     i_list = [i]
        #     result_image = kp.apply_keypoints_mask(disparity, i_list)
        #     result_image_list.append(result_image)
           
        
        #save_image("../images/prediction_results/", result_image, "filtered_keypoints", False)

        eps = 50 if "matlab" in camera_type else 10
        min_samples = 6

    for kp_xy in keypoints:
        # --------------------------------------------------------------------------
        # Se agregó keypoints para no usar la mascara porque se pierde la posición de los keypoints
        point_cloud, colors = disparity_to_pointcloud(disparity, Q, img_l, kp_xy)
        # point_cloud = point_cloud.astype(np.float64)
        point_cloud_list.append(point_cloud), colors_list.append(colors)
    
    return point_cloud_list, colors_list, eps, min_samples, keypoints

def roi_no_dense_pc(img_l, disparity, Q):
    segmentation = kp.get_segmentation(img_l)
    point_clouds, pc_colors = [], []

    for seg in segmentation:
        mask = kp.apply_seg_individual_mask(img_l, seg)
        #save_image("../images/prediction_results/prueba", mask, "individuo", False)
        point_cloud, color = disparity_to_pointcloud(disparity, Q, img_l, mask)
        point_cloud = point_cloud.astype(np.float64)
        point_clouds.append(point_cloud)
        pc_colors.append(color)

    return point_clouds, pc_colors 
    

def roi_source_point_cloud(img_l, img_r, Q):
    eps, min_samples = 5, 1800

    roi_left = kp.get_roi(img_l)
    roi_right = kp.get_roi(img_r)

    result_img_left = kp.apply_roi_mask(img_l, roi_left)
    result_img_right = kp.apply_roi_mask(img_r, roi_right)

    disparity = compute_disparity(result_img_left, result_img_right)

    filtered_disparity = kp.apply_roi_mask(disparity, roi_left)

    dense_point_cloud, dense_colors = disparity_to_pointcloud(disparity, Q, img_l, filtered_disparity)

    return filtered_disparity, dense_point_cloud, dense_colors, eps, min_samples


def point_cloud_correction(points, model):
    points = np.asarray(points)

    Xy = points[:,:2]
    # Predecir las coordenadas Z corregidas usando el modelo
    z = points[:, 2].reshape(-1,1)  # Tomar solo las coordenadas X, y
    z_pred = model.predict(z)
    
    # Actualizar las coordenadas Z de la nube de puntos con las predicciones corregidas
    corrected_points = np.column_stack((Xy, z_pred))
    return corrected_points


def save_dense_point_cloud(point_cloud, colors, base_filename):
    if not os.path.exists(os.path.dirname(base_filename)):
        os.makedirs(os.path.dirname(base_filename))
    dense_filename = f"{base_filename}_dense.ply"
    save_point_cloud(point_cloud, colors, dense_filename)


