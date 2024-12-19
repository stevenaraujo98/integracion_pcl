import numpy as np
from scipy.spatial import KDTree

def depth_to_xyz(depth_image, depth_scale, intrinsics):  
    """  
    Convierte una imagen de profundidad a coordenadas 3D (X, Y, Z).  

    Args:  
        depth_image (np.ndarray): Imagen de profundidad (en píxeles).  
        depth_scale (float): Escala de profundidad (en metros por unidad).  
        intrinsics (rs.intrinsics): Parámetros intrínsecos de la cámara.  

    Returns:  
        np.ndarray: Nube de puntos en coordenadas 3D (forma: [H*W, 3]).  
    """  
    # Obtener las intrínsecas de la cámara  
    fx, fy = intrinsics.fx, intrinsics.fy  
    cx, cy = intrinsics.ppx, intrinsics.ppy  

    # Obtener las dimensiones de la imagen  
    height, width = depth_image.shape  

    # Crear una cuadrícula de coordenadas de píxeles (u, v)  
    u, v = np.meshgrid(np.arange(width), np.arange(height))  

    # Convertir la profundidad a metros  
    Z = depth_image * depth_scale  

    # Calcular las coordenadas 3D  
    X = (u - cx) * Z / fx  
    Y = (v - cy) * Z / fy  

    # Apilar las coordenadas en un array de forma (H*W, 3)  
    points_3d = np.dstack((X, Y, Z)).reshape(-1, 3)  

    return points_3d

def xy_to_xyz(xy_coords, depth_image, depth_scale, intrinsics, to_unit="m"):
    """
    Convierte coordenadas 2D (x, y) y una imagen de profundidad a coordenadas 3D (X, Y, Z).

    Args:
        xy_coords (list or np.ndarray): Lista o array de coordenadas (x, y) en píxeles.
        depth_image (np.ndarray): Imagen de profundidad (en píxeles).
        depth_scale (float): Escala de profundidad (en metros por unidad).
        intrinsics (rs.intrinsics): Parámetros intrínsecos de la cámara.

    Returns:
        np.ndarray: Coordenadas 3D (X, Y, Z) correspondientes a las coordenadas (x, y).
    """
    if to_unit == "m":
        factor = 1  # De metros a milímetros
    elif to_unit == "mm":
        factor = 1000  # De metros a milímetros
    elif to_unit == "cm":
        factor = 100  # De metros a centímetros
    else:
        raise ValueError("Unidad no válida. Usa 'mm' para milímetros o 'cm' para centímetros.")

    # Obtener las intrínsecas de la cámara
    fx, fy = intrinsics.fx, intrinsics.fy
    cx, cy = intrinsics.ppx, intrinsics.ppy

    # Lista para almacenar las coordenadas 3D
    points_3d = []

    # Iterar sobre las coordenadas (x, y)
    for x, y in xy_coords:
        # dentro de los límites de la imagen
        if 0 < y < depth_image.shape[0] and 0 < x < depth_image.shape[1]:
            # Obtener la profundidad en la posición (x, y)
            Z = depth_image[int(y), int(x)] * depth_scale 

            # Calcular las coordenadas 3D
            X = (x - cx) * Z / fx
            Y = (y - cy) * Z / fy

            points_3d.append([X * factor, Y * factor, Z * factor])
        else:
            points_3d.append([0, 0, 0])

    # Convertir la lista a un array de NumPy
    return np.array(points_3d)


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
