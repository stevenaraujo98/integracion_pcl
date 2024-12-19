import pyrealsense2 as rs
import numpy as np
import cv2
import math
from ultralytics import YOLO
from dense.keypoint_extraction import get_keypoints, apply_keypoints_mask
import open3d as o3d

class AppState:

    def __init__(self, *args, **kwargs):
        self.WIN_NAME = 'RealSense'
        self.pitch, self.yaw = math.radians(-10), math.radians(-15)
        self.translation = np.array([0, 0, -1], dtype=np.float32)
        self.distance = 2
        self.prev_mouse = 0, 0
        self.mouse_btns = [False, False, False]
        self.paused = False
        self.decimate = 1
        self.scale = True
        self.color = True

    def reset(self):
        self.pitch, self.yaw, self.distance = 0, 0, 2
        self.translation[:] = 0, 0, -1

    @property
    def rotation(self):
        Rx, _ = cv2.Rodrigues((self.pitch, 0, 0))
        Ry, _ = cv2.Rodrigues((0, self.yaw, 0))
        return np.dot(Ry, Rx).astype(np.float32)

    @property
    def pivot(self):
        return self.translation + np.array((0, 0, self.distance), dtype=np.float32)

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

state = AppState()

# Create a pipeline
pipeline = rs.pipeline()

# Create a config and configure the pipeline to stream
#  different resolutions of color and depth streams
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))
print("device_product_line", device_product_line)

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

# ----- Lidar
# config.enable_stream(rs.stream.depth, 1024, 768, rs.format.z16, 30) # 23.6M 70o x 55o 0.25 - 2.6m 0.25 - 6.5m
# config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30) # 9.2M 70o x 55o 0.25 - 3.9m 0.25 - 9m
# config.enable_stream(rs.stream.depth, 320, 240, rs.format.z16, 30) # 2.3M 70o x 55o 0.25 - 3.9m 0.25 - 9m

# config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30) # 6,15,30
# config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30) # 6,15,30,60
# config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30) # 6,15,30,60

# Depth
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30) # 6, 15, 30
# config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30) # 6, 15, 30, 60, 90
# config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30) # 6, 15, 30, 60, 90
# config.enable_stream(rs.stream.depth, 640, 360, rs.format.z16, 30) # 6, 15, 30, 60, 90
# config.enable_stream(rs.stream.depth, 480, 720, rs.format.z16, 30) # 6, 15, 30, 60, 90
# config.enable_stream(rs.stream.depth, 424, 240, rs.format.z16, 30) # 6, 15, 30, 60, 90

config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30) # 6,15,30
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30) # 6,15,30
# config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30) # 6, 15, 30, 60
# config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30) # 6, 15, 30, 60
# config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30) # 6, 15, 30, 60
# config.enable_stream(rs.stream.color, 640, 360, rs.format.bgr8, 30) # 6, 15, 30, 60
# config.enable_stream(rs.stream.color, 424, 240, rs.format.bgr8, 30) # 6, 15, 30, 60
# config.enable_stream(rs.stream.color, 320, 240, rs.format.bgr8, 30) # 6, 30, 60
# config.enable_stream(rs.stream.color, 320, 180, rs.format.bgr8, 30) # 6, 30, 60

# Start streaming
profile = pipeline.start(config)

# Getting the depth sensor's depth scale (see rs-align example for explanation)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: " , depth_scale)

# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
align_to = rs.stream.color
align = rs.align(align_to)

# -------------------------------------------------------------------------------------------------
points = np.random.rand(17, 3)  # 100 puntos aleatorios en 3D

# Crear una nube de puntos en Open3D
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(points)
vis = o3d.visualization.Visualizer()
vis.create_window()
vis.add_geometry(point_cloud)
# -------------------------------------------------------------------------------------------------


# Streaming loop
try:
    while True:
        # Get frameset of color and depth
        frames = pipeline.wait_for_frames()

        # Align the depth frame to color frame
        aligned_frames = align.process(frames)

        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame 
        color_frame = aligned_frames.get_color_frame()

        # Grab new intrinsics (may be changed by decimation)
        depth_intrinsics = rs.video_stream_profile( aligned_depth_frame.profile ).get_intrinsics()
        
        # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            continue

        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        print("depth_image", depth_image.shape, "color_image", color_image.shape, "aligned_depth_frame", aligned_depth_frame.get_width(), aligned_depth_frame.get_height())

        keypoints = get_keypoints(color_image)
        print("keypoints", keypoints)
        result_image = apply_keypoints_mask(depth_image, keypoints)

        color_image_copy = color_image.copy()
        # color_image_copy = color_image_copy[color_image_copy.shape[0]//2:, color_image_copy.shape[1]//2:]

        depth_image_copy = result_image.copy()
        # depth_image_copy = depth_image_copy[depth_image_copy.shape[0]//2:, depth_image_copy.shape[1]//2:]

        filtered_array = np.array([])
        if len(keypoints[0]) > 0:
            for kps in keypoints:
                for kp in kps:
                    cv2.circle(color_image_copy, (int(kp[0]), int(kp[1])), 5, (255, 255, 255), -1) 
        
            points_3d = depth_to_xyz(depth_image_copy, depth_scale, depth_intrinsics)
            filtered_array = points_3d[~np.all(points_3d == 0, axis=1)]
            print(filtered_array)

            # Calculo del centroide
            centroid = np.mean(filtered_array, axis=0)
            centered_points = filtered_array - centroid

            # Actualizar la nube de puntos en Open3D
            point_cloud.points = o3d.utility.Vector3dVector(centered_points)

            # # Asignar un color uniforme (por ejemplo, azul)
            # colors = np.array([[0, 0, 1] for _ in range(centered_points.shape[0])])  # Azul (R=0, G=0, B=1)
            # point_cloud.colors = o3d.utility.Vector3dVector(colors)

            # Configurar la vista de la cámara
            ctr = vis.get_view_control()
            ctr.set_zoom(0.8)  # Ajusta el nivel de zoom
            # ctr.set_front([0, 0, -1])  # Dirección de la cámara
            ctr.set_lookat([0, 0, 0])  # Punto de enfoque (centrado en el origen)
            ctr.set_up([0, -1, 0])  # Dirección "arriba"

            vis.update_geometry(point_cloud)
            vis.poll_events()
            vis.update_renderer()


        cv2.imshow('depth_image', depth_image_copy)
        cv2.imshow('color_image', color_image_copy)

        key = cv2.waitKey(1)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
finally:
    pipeline.stop()


"""
source /home/admin-cidis/anaconda3/bin/activate
conda create -n lidar python=3.8
conda activate lidar2

nvcc --version
"""