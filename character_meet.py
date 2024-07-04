import numpy as np
import cv2

# Normalizar y escalar puntos
def normalize_and_scale(point, min_x, range_x, min_y, range_y, width, height):
    x = (point[0] - min_x) / range_x * width
    y = (point[1] - min_y) / range_y * height
    return int(x), int(height - y)  # Invertir y para que el origen esté en la esquina inferior izquierda

# def normalize(x, y, max_x, max_y, width, height, point):
    


def get_img_shape_meet(list_centroides):
    image_white = np.ones((640, 640, 3), dtype=np.uint8) * 255

    # unir los puntos con una línea
    list_xz = []
    for i in list_centroides:
        list_xz.append((i[0], i[2]))
    
    # Encontrar los mínimos y máximos de las coordenadas x y y
    min_x = min(point[0] for point in list_xz)
    max_x = max(point[0] for point in list_xz)
    min_y = min(point[1] for point in list_xz)
    max_y = max(point[1] for point in list_xz)

    # Asegurarse de que no se divida por cero si min_x == max_x o min_y == max_y
    range_x = max_x - min_x if max_x != min_x else 1
    range_y = max_y - min_y if max_y != min_y else 1

    # Convertir los puntos a coordenadas de imagen
    scaled_points = [normalize_and_scale(point, min_x, range_x, min_y, range_y, 640, 640) for point in list_xz]
    # scaled_points = [normalize(min_x, min_y, max_x, max_y, point, 640, 640) for point in list_xz]

    # Dibujar líneas entre los puntos
    for i in range(len(scaled_points) - 1):
        cv2.line(image_white, scaled_points[i], scaled_points[i + 1], (0, 0, 0), 2)

    
    # Convertir la imagen a escala de grises
    gray_image = cv2.cvtColor(image_white, cv2.COLOR_BGR2GRAY)
    
    cv2.imshow("Grayscale Image", gray_image)
    cv2.waitKey(0)
    
    return gray_image

def get_img_shape_meet_prev_sort(list_centroides):
    image_white = np.ones((640, 640, 3), dtype=np.uint8) * 255

    if len(list_centroides) == 0:
        return image_white
    
    # unir los puntos con una línea
    list_xz = []
    """
    for i in list_centroides:
        list_xz.append([(int(i[0][0])+320, int(i[0][2])), (int(i[1][0])+320, int(i[1][2]))])

    # Dibujar líneas entre los puntos
    for i in list_xz:
        cv2.line(image_white, i[0], i[1], (0, 0, 0), 2)
    """

    ## todo lo que va a normalizarse
    list_points = []
    for i in list_centroides:
        list_xz.append([(int(i[0][0]), int(i[0][2])), (int(i[1][0]), int(i[1][2]))])
        list_points.append((int(i[0][0]), int(i[0][2])))
        list_points.append((int(i[1][0]), int(i[1][2])))

    # Encontrar los mínimos y máximos de las coordenadas x y y
    min_x = min(point[0] for point in list_points)
    max_x = max(point[0] for point in list_points)
    min_y = min(point[1] for point in list_points)
    max_y = max(point[1] for point in list_points)

    # Asegurarse de que no se divida por cero si min_x == max_x o min_y == max_y
    range_x = max_x - min_x if max_x != min_x else 1
    range_y = max_y - min_y if max_y != min_y else 1

    # Convertir los puntos a coordenadas de imagen
    scaled_points = [(normalize_and_scale(points[0], min_x, range_x, min_y, range_y, 640, 640), normalize_and_scale(points[1], min_x, range_x, min_y, range_y, 640, 640)) for points in list_xz]

    # Dibujar líneas entre los puntos
    for i in scaled_points:
        cv2.line(image_white, i[0], i[1], (0, 0, 0), 2)
    
    # Convertir la imagen a escala de grises
    gray_image = cv2.cvtColor(image_white, cv2.COLOR_BGR2GRAY)

    print("Save gray_image", "images/shape/gray_image.jpg")
    # save gray_image
    cv2.imwrite("images/shape/gray_image.jpg", gray_image)

    
    # cv2.imshow("Grayscale Image", gray_image)
    # cv2.waitKey(0)
    
    return gray_image