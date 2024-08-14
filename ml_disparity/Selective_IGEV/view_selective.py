import string
import plotly.graph_objects as go
import numpy as np
from pathlib import Path
import cv2 as cv
import os
import open3d as o3d
from bridge_selective import get_SELECTIVE_disparity_map
import util

# Parámetros de la cámara
fx, fy, cx1, cy = 1429.4995220185822, 1430.4111785502332, 929.8227256572083, 506.4722541384677
cx2 = 936.8035788332203
baseline = 32.95550620237698 # en milímetros


pairs = util.read_image_pairs_by_distance("./datasets/CIDIS/validation")
alphabet = string.ascii_lowercase

output_directory = "demo_output"
# Genera el mapa de disparidad usando la función de bridge.py


# parece ser que el error del cuadrado en el mapa de disparidad descrito en las ppts es debido al entrenamiento de los modelos, este error parece suceder cuando 
# los modelos han sido entrenados con imagenes cercanas.
restore_ckpt = Path("pretrained_models/middlebury_train.pth")

counter = 0
for situation, variations in pairs.items():
    for variation, letter in zip(variations, alphabet):
        left_imgs = variation[0]
        right_imgs = variation[1]
        
        
        print(f"Iteración {counter + 1}")

        

        # # Define la ruta del directorio testF
        # testF_folder = "./datasets/CIDIS/Laser_ground_truth/790"
        # image_path = testF_folder + "/" +"15_30_41_07_06_2024_IMG_LEFT.jpg"
        # left_imgs = str(image_path)
        # right_imgs = str(testF_folder + "/" + "15_30_41_07_06_2024_IMG_RIGHT.jpg")


        
        disp = get_SELECTIVE_disparity_map(
    restore_ckpt=restore_ckpt,
    left_imgs=left_imgs,
    right_imgs=right_imgs,
    save_numpy=True
)

        print(disp.shape)
        # Verifica la forma de disp
        print(f"Forma de disp: {disp.shape}")

        # Asegúrate de que disp sea 2D
        if disp.ndim > 2:
            disp = np.squeeze(disp)  # Remueve todas las dimensiones de tamaño 1

        print(f"Forma de disp después de squeeze: {disp.shape}")

        # Verifica si la imagen existe antes de cargarla
        if left_imgs != None:
            image = cv.imread(left_imgs, cv.IMREAD_COLOR)
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        else:
            print(f"La imagen {left_imgs} no existe.")
            exit(1)

        # Proyección inversa
        depth = (fx * baseline) / (-disp + (cx2 - cx1))

        # Verifica la forma de depth
        print(f"Forma de depth: {depth.shape}")

        # Asegúrate de que depth sea 2D
        if depth.ndim > 2:
            depth = np.squeeze(depth)  # Remueve todas las dimensiones de tamaño 1

        # Verifica que depth tenga dos dimensiones
        if depth.ndim != 2:
            raise ValueError(f"depth debe tener dos dimensiones, pero tiene {depth.ndim} dimensiones")

        H, W = depth.shape
        print(f"H: {H}, W: {W}")

        xx, yy = np.meshgrid(np.arange(W), np.arange(H), indexing='xy')
        print(f"Forma de xx: {xx.shape}, Forma de yy: {yy.shape}")

        points_grid = np.stack(((xx - cx1) / fx, (yy - cy) / fy, np.ones_like(xx)), axis=-1) * depth[:, :, np.newaxis]
        print(f"Forma de points_grid después de multiplicación: {points_grid.shape}")

        # Filtra los puntos voladores (opcional)
        mask = np.ones((H, W), dtype=bool)
        mask[1:][np.abs(depth[1:] - depth[:-1]) > 1] = False
        mask[:,1:][np.abs(depth[:,1:] - depth[:,:-1]) > 1] = False

        points = points_grid[mask].astype(np.float64)
        colors = image[mask].astype(np.float64) / 255

        # Invertir la coordenada X para corregir la orientación
        points[:, 0] = -points[:, 0]

        print(f"Situacion = {situation}")
        # Visualización de la nube de puntos en 3D

        print("""
        Controls:
        ---------
        Zoom:      Scroll Wheel
        Translate: Right-Click + Drag
        Rotate:    Left-Click + Drag
        """)

        # # OPEN3D
        # origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10, origin=[0,0,0])
        # # Crear la nube de puntos
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(points)
        # pcd.colors = o3d.utility.Vector3dVector(colors)

        # # Crear el visualizador
        # viewer = o3d.visualization.Visualizer()
        # viewer.create_window(window_name="Nube de Puntos 3D", width=800, height=600)

        # # Agregar la geometría
        # viewer.add_geometry(pcd)
        # viewer.add_geometry(origin)

        # # Configurar opciones de renderizado
        # opt = viewer.get_render_option()
        # opt.point_size = 1  # Ajustar el tamaño de los puntos

        # # Renderizar y visualizar
        # viewer.run()
        # viewer.destroy_window()



        # Añadir el origen (0,0,0) con un color distintivo (rojo)
        origin_point = np.array([[0, 0, 0]])
        origin_color = np.array([[1, 0, 0]])  # Color rojo

        # Combinar el origen con los puntos y colores existentes
        points = np.vstack([points, origin_point])
        colors = np.vstack([colors, origin_color])

        # Seleccionar un subconjunto aleatorio de puntos, incluyendo el origen
        NUM_POINTS_TO_DRAW = 250000
        subset = np.random.choice(points.shape[0], size=(NUM_POINTS_TO_DRAW - 1,), replace=False)
        subset = np.append(subset, points.shape[0] - 1)  # Asegurar que el origen esté incluido
        points_subset = points[subset]
        colors_subset = colors[subset]

        x, y, z = points_subset.T

        fig = go.Figure(
            data=[
                go.Scatter3d(
                    x=x, y=y, z=z, # flipped to make visualization nicer
                    mode='markers',
                    marker=dict(size=1, color=colors_subset)
                )
            ],
            layout=dict(
                scene=dict(
                    xaxis=dict(visible=True),
                    yaxis=dict(visible=True),
                    zaxis=dict(visible=True),
                )
            )
        )
        fig.show()
        if (counter + 1) % 4 == 0:
            util.esperar_tecla("q")

        counter+= 1
        next