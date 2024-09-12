import open3d as o3d
import numpy as np
import dense.pc_generation as pcGen
import dense.pc_generation_ML as pcGen_ML
from scipy.spatial import cKDTree
from scipy.spatial import KDTree
from sklearn.cluster import DBSCAN
from ml_disparity.Selective_IGEV.bridge_selective import get_SELECTIVE_disparity_map
from ml_disparity.RAFTStereo.bridge_raft import get_RAFT_disparity_map
import json
import os
import cv2

def compute_disparity(img_left: np.array, img_right: np.array, config: dict, method: str):
    """
    Calcula el mapa de disparidad de un par de imágenes usando el método especificado en la configuración.
    
    :param img_left: Imagen del lado izquierdo como array de numpy.
    :param img_right: Imagen del lado derecho como array de numpy.
    :param config: Diccionario de configuración para un perfil específico.
    :param method: Método de disparidad a utilizar (e.g., 'SGBM', 'WLS-SGBM', 'RAFT', 'SELECTIVE').
    :return: Mapa de disparidad como array de numpy.
    """
    # Acceso a los métodos de disparidad configurados
    methods_config = config['disparity_methods']
    
    if method == 'SGBM' and methods_config['SGBM']['enabled']:
        params = methods_config['SGBM']['params']
        disparity = pcGen.compute_disparity(img_left, img_right, params)

    elif method == 'WLS-SGBM' and methods_config['WLS-SGBM']['enabled']:
        params = methods_config['WLS-SGBM']['params']
        disparity = pcGen.compute_disparity(img_left, img_right, params)

    elif method == 'RAFT' and methods_config['RAFT']['enabled']:
        disparity = get_RAFT_disparity_map(
            restore_ckpt=methods_config['RAFT']['params']['restore_ckpt'],
            img_left_array=img_left,
            img_right_array=img_right,
            save_numpy=True, 
            slow_fast_gru=True, #--slow_fast_gru: True=GRUs de baja resolución iteran más frecuentemente (captura cambios rápidos, más tiempo de computación), 
                                #                 False(Default)=frecuencia estándar (suficiente para muchas aplicaciones, más eficiente).
        )
        disparity = disparity[0]
    elif method == 'SELECTIVE' and methods_config['SELECTIVE']['enabled']:
        # Usar Selective para calcular disparidad
        disparity = get_SELECTIVE_disparity_map(
            restore_ckpt=methods_config['SELECTIVE']['params']['restore_ckpt'],
            img_left_array=img_left,
            img_right_array=img_right,
            save_numpy=True,
            slow_fast_gru=True,
        )
        disparity = disparity[0]
    else:
        raise ValueError(f"The disparity method {method} is either not enabled or does not exist in the configuration.")

    return disparity

# Clase encargada de la normalizacion estandar de las nubes de puntos
class PointCloudScaler:
    def __init__(self, reference_point, scale_factor):
        self.reference_point = np.array(reference_point)
        self.scale_factor = scale_factor

    def calculate_scaled_positions(self, points):
        # Restar el punto de referencia a todos los puntos
        shifted_points = points - self.reference_point
        
        # Escalar los puntos
        scaled_points = self.scale_factor * shifted_points
        
        # Volver a mover los puntos al sistema de referencia original
        new_positions = scaled_points + self.reference_point
        
        return new_positions

    def scale_cloud(self, points):
        # Procesa todos los puntos sin dividirlos en trozos ni usar procesamiento paralelo
        new_positions = self.calculate_scaled_positions(points)
        return new_positions

def correct_depth_o3d(points, alpha=0.5):
    """
    Aplica una corrección de profundidad a una nube de puntos 3D numpy array.
    
    :param points: Numpy array de puntos 3D
    :param alpha: Parámetro de la transformación de potencia (0 < alpha < 1)
    :return: Numpy array de puntos 3D corregidos
    """
    X, Y, Z = points[:, 0], points[:, 1], points[:, 2]
    Z_safe = np.where(Z == 0, np.finfo(float).eps, Z)
    Z_corrected = Z_safe ** alpha
    X_corrected = X * (Z_corrected / Z_safe)
    Y_corrected = Y * (Z_corrected / Z_safe)
    corrected_points = np.vstack((X_corrected, Y_corrected, (0.6947802265318861*Z_corrected) + -14.393348239171985)).T
    return corrected_points

def process_numpy_point_cloud(points_np, reference_point=[0, 0, 0], scale_factor=0.280005, alpha=1.0005119):
    # Escalar la nube de puntos
    scaler = PointCloudScaler(reference_point=reference_point, scale_factor=scale_factor)
    scaled_points_np = scaler.scale_cloud(points_np)
    
    # Aplicar corrección de profundidad
    corrected_points_np = correct_depth_o3d(scaled_points_np, alpha)
    
    return corrected_points_np



def generate_individual_filtered_point_clouds(img_left: np.array, img_right: np.array, config: dict, method: str, use_roi: bool, use_max_disparity: bool, normalize: bool = True):
    """
    Genera y retorna listas separadas de nubes de puntos y colores para cada objeto detectado individualmente, utilizando un método específico de disparidad y configuraciones de filtrado avanzadas.

    :param img_left: Imagen del lado izquierdo como array de numpy.
    :param img_right: Imagen del lado derecho como array de numpy.
    :param config: Diccionario de configuración para un perfil específico.
    :param method: Método de disparidad a utilizar (e.g., 'SGBM', 'RAFT', 'SELECTIVE').
    :param use_roi: Booleano que indica si se debe aplicar una Región de Interés (ROI) durante el procesamiento.
    :param use_max_disparity: Booleano que indica si se debe utilizar la disparidad máxima para optimizar la nube de puntos.
    :return: Listas de nubes de puntos y colores, cada una correspondiente a un objeto detectado individualmente.
    """
    # Generar el mapa de disparidad utilizando la función adecuada
    disparity_map = compute_disparity(img_left, img_right, config, method)

    # Acceder a parámetros relevantes desde la configuración
    Q = np.array(config['camera_params']['Q_matrix'])
    fx = config['camera_params']['fx']
    fy = config['camera_params']['fy']
    cx1 = config['camera_params']['cx1']
    cx2 = config['camera_params']['cx2']
    cy = config['camera_params']['cy']
    baseline = config['camera_params']['baseline']

    # Generar nubes de puntos filtradas para cada objeto detectado
    if method == 'SGBM' or method == 'WLS-SGBM':
        point_cloud_list, color_list, eps, min_samples, keypoints3d_list, res_kp_seg = pcGen.generate_filtered_point_cloud(
            img_left, disparity_map, Q, "matlab", use_roi, use_max_disparity,
            
        )
        scale_factor = 3.45
    else:
        point_cloud_list, color_list, eps, min_samples, keypoints3d_list, res_kp_seg = pcGen_ML.generate_filtered_point_cloud(
            img_left, disparity_map, fx, fy, cx1, cx2, cy, baseline,"matlab", use_roi, use_max_disparity,
           
        )
        scale_factor = 0.280005

    # Normalizar la nube de puntos si se solicita
    if normalize:
        normalized_point_cloud_list = [process_numpy_point_cloud(cloud, scale_factor=scale_factor, alpha=1.0005119) for cloud in point_cloud_list]
        normalized_keypoints_list = [process_numpy_point_cloud(kps, scale_factor=scale_factor, alpha=1.0005119) for kps in keypoints3d_list]
        return normalized_point_cloud_list, color_list, normalized_keypoints_list, res_kp_seg

    return point_cloud_list, color_list, keypoints3d_list, res_kp_seg

def load_config(path):
    """
    Carga la configuración desde un archivo JSON.
    """
    with open(path, 'r') as file:
        config = json.load(file)
    return config


def load_stereo_maps(xml_file: str):
    """
    Carga los mapas de rectificación estéreo desde un archivo XML.

    Args:
        xml_file (str): Ruta al archivo XML que contiene los mapas de rectificación.

    Returns:
        dict: Un diccionario con los mapas de rectificación para las imágenes izquierda y derecha.
    """
    cv_file = cv2.FileStorage(xml_file, cv2.FILE_STORAGE_READ)
    stereoMapL_x = cv_file.getNode('stereoMapL_x').mat()
    stereoMapL_y = cv_file.getNode('stereoMapL_y').mat()
    stereoMapR_x = cv_file.getNode('stereoMapR_x').mat()
    stereoMapR_y = cv_file.getNode('stereoMapR_y').mat()
    cv_file.release()

    return {
        'Left': (stereoMapL_x, stereoMapL_y),
        'Right': (stereoMapR_x, stereoMapR_y)
    }   

def rectify_images(img_left: np.array, img_right: np.array, config: str):
    """
    Rectifica un par de imágenes estéreo usando los mapas de rectificación correspondientes al perfil de calibración dado.

    Args:
        img_left (np.array): Imagen izquierda como array de numpy.
        img_right (np.array): Imagen derecha como array de numpy.
        profile_name (str): Nombre del perfil que contiene los mapas de rectificación.

    Returns:
        tuple: Tupla que contiene las imágenes izquierda y derecha rectificadas.
    """
    # Carga los mapas de rectificación desde el archivo XML asociado al perfil
    map_path = f'dense/config_files/{config}/stereo_map.xml'
    if not os.path.exists(map_path):
        raise FileNotFoundError("No se encontró el archivo de mapa de rectificación para el perfil especificado.")
    
    stereo_maps = load_stereo_maps(map_path)
    
    # Aplica los mapas de rectificación
    img_left_rect = cv2.remap(img_left, stereo_maps['Left'][0], stereo_maps['Left'][1], cv2.INTER_LINEAR)
    img_right_rect = cv2.remap(img_right, stereo_maps['Right'][0], stereo_maps['Right'][1], cv2.INTER_LINEAR)

    img_left_rect = cv2.cvtColor(img_left_rect, cv2.COLOR_BGR2RGB)
    img_right_rect = cv2.cvtColor(img_right_rect, cv2.COLOR_BGR2RGB)
    return img_left_rect, img_right_rect



def compute_centroid(points, k=5, threshold_factor=1.0):
    if len(points) < k + 1:
        raise ValueError("La nube de puntos es demasiado pequeña para el valor de k.")

    # Construir un KDTree para la búsqueda de vecinos
    tree = KDTree(points)
    noise_mask = np.zeros(points.shape[0], dtype=bool)

    # Calcular la distancia euclidiana promedio entre todos los puntos
    distances, _ = tree.query(points, k=k+1)
    avg_distance = np.mean(distances[:, 1:])  # Omitir la distancia al propio punto
    threshold = threshold_factor * avg_distance

    for i, point in enumerate(points):
        # Encontrar los k vecinos más cercanos
        distances, indices = tree.query(point, k=k+1)
        # Omitir el propio punto
        distances = distances[1:]
        indices = indices[1:]

        # Calcular la profundidad media
        mean_depth = np.mean(distances)

        # Marcar el punto como ruido si su profundidad excede el umbral
        if np.any(np.abs(distances - mean_depth) > threshold):
            noise_mask[i] = True

    # Filtrar los puntos no ruidosos
    filtered_points = points[~noise_mask]

    if len(filtered_points) == 0:
        raise ValueError("Todos los puntos fueron considerados como ruido.")

    # Calcular el centroide de los puntos no ruidosos
    centroid = np.mean(filtered_points, axis=0)

    return centroid

def filter_points_by_optimal_range(point_cloud, centroid, m_initial=30):
    z_centroid = centroid[2]
    m = m_initial  # Puedes ajustar esto si necesitas una relación más compleja
    lower_bound = z_centroid - m
    upper_bound = z_centroid + m

    # Crear una máscara lógica para filtrar los puntos en el rango óptimo
    mask = (point_cloud[:, 2] >= lower_bound) & (point_cloud[:, 2] <= upper_bound)
    filtered_points = point_cloud[mask]

    return filtered_points

def get_Y_bounds(filtered_points):
    if filtered_points.size == 0:
        return None, None

    y_min = np.min(filtered_points[:, 1])
    y_max = np.max(filtered_points[:, 1])

    return y_min, y_max

def estimate_height_from_point_cloud(point_cloud: np.array, k: int = 5, threshold_factor: float = 1.0, m_initial: float = 50.0):
    """
    Estima la altura de una persona a partir de una nube de puntos y calcula el centroide de los keypoints.

    Args:
        point_cloud (np.array): Nube de puntos 3D representada como un array numpy de forma (N, 3).
        k (int): Número de vecinos más cercanos para calcular el centroide.
        threshold_factor (float): Factor de umbral para eliminar el ruido en el cálculo del centroide.
        m_initial (float): Rango inicial para filtrar los puntos alrededor del centroide.

    Returns:
        Tuple[float, np.array]: Altura estimada de la persona y el centroide calculado.
    """
    try:
        # Calcular el centroide de la nube de puntos
        centroid = compute_centroid(point_cloud, k=k, threshold_factor=threshold_factor)

        # Filtrar los puntos de la nube en un rango óptimo basado en el centroide
        filtered_points = filter_points_by_optimal_range(point_cloud, centroid, m_initial)

        # Obtener los límites mínimos y máximos en Y (altura)
        y_min, y_max = get_Y_bounds(filtered_points)
        if y_min is not None and y_max is not None:
            # Calcular la altura como la diferencia entre Y_max y Y_min
            height = abs(y_max - y_min)
            # print(f"Para el centroide con z = {centroid[2]}, el rango de Y es: Y_min = {y_min}, Y_max = {y_max}")
            # print(f"La altura de la persona es de {height}\n")
            return height, centroid
        else:
            print("No se encontraron puntos en el rango óptimo para este centroide.")
            return None, centroid

    except ValueError as ve:
        print(f"Error al calcular el centroide: {ve}")
        return None, None