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

# def get_img_shape_meet_prev_sort(list_centroides, name_common, step_frames, centroide):
#     image_white = np.ones((640, 640, 3), dtype=np.uint8) * 255

#     if len(list_centroides) == 0:
#         return image_white
    
#     # unir los puntos con una línea
#     list_xz = []
#     """
#     for i in list_centroides:
#         list_xz.append([(int(i[0][0])+320, int(i[0][2])), (int(i[1][0])+320, int(i[1][2]))])

#     # Dibujar líneas entre los puntos
#     for i in list_xz:
#         cv2.line(image_white, i[0], i[1], (0, 0, 0), 2)
#     """

#     ## todo lo que va a normalizarse
#     list_points = []
#     for i in list_centroides:
#         list_xz.append([(int(i[0][0]), int(i[0][2])), (int(i[1][0]), int(i[1][2]))])
#         list_points.append((int(i[0][0]), int(i[0][2])))
#         list_points.append((int(i[1][0]), int(i[1][2])))

#     # Encontrar los mínimos y máximos de las coordenadas x y y
#     min_x = min(point[0] for point in list_points)
#     max_x = max(point[0] for point in list_points)
#     min_y = min(point[1] for point in list_points)
#     max_y = max(point[1] for point in list_points)

#     # Asegurarse de que no se divida por cero si min_x == max_x o min_y == max_y
#     range_x = max_x - min_x if max_x != min_x else 1
#     range_y = max_y - min_y if max_y != min_y else 1

#     # Convertir los puntos a coordenadas de imagen
#     scaled_points = [(
#         normalize_and_scale(points[0], min_x, range_x, min_y, range_y, 640, 640), 
#         normalize_and_scale(points[1], min_x, range_x, min_y, range_y, 640, 640)
#     ) for points in list_xz]

#     # Dibujar líneas entre los puntos
#     for i in scaled_points:
#         cv2.line(image_white, i[0], i[1], (0, 0, 0), 2)
    
#     # Convertir la imagen a escala de grises
#     gray_image = cv2.cvtColor(image_white, cv2.COLOR_BGR2GRAY)

#     print("Save gray_image", "images/shape/gray_image_" + str(name_common) + str(step_frames) + ".jpg")
#     # save gray_image
#     cv2.imwrite("images/shape/gray_image_" + str(name_common) + str(step_frames) + ".jpg", gray_image)

    
#     # cv2.imshow("Grayscale Image", gray_image)
#     # cv2.waitKey(0)
    
#     return gray_image

def draw_line_with_conditions(image, pt1, pt2, mean_x, val_f):
    x1, y1 = pt1
    x2, y2 = pt2
    _, width = image.shape[:2]
    x_2 = 0

    # Determinar la dirección de la línea
    if x1 < mean_x and x1 < x2 and val_f:
        start, end = (x2, y2), (x1, y1)
        x_2 = 0
        val_f = False
    elif x2 < mean_x and x2 < x1 and val_f:
        start, end = (x1, y1), (x2, y2)
        x_2 = 0
        val_f = False
    elif x1 > mean_x and x1 > x2:
        start, end = (x2, y2), (x1, y1)
        x_2 = width-1
    elif x2 > mean_x and x2 > x1:
        start, end = (x1, y1), (x2, y2)
        x_2 = width-1
    
    # Calcular la pendiente y el intercepto
    # y = m x + b
    m = (end[1] - start[1]) / (end[0] - start[0])
    b = start[1] - m * start[0]
    # print("m", m, "b", b)
    
    y_res = int(m * x_2 + b)

    cv2.line(image, start, (x_2, y_res), (0, 0, 0), 2)

    return val_f


def get_img_shape_meet_prev_sort(list_centroides, name_common, step_frames, centroide):
    big_size = 800
    re_size = 640
    half_re_size = re_size//2
    # generar la imagen hasta el limite maximo
    img = np.ones((big_size, big_size, 3), dtype=np.uint8) * 255

    # # graficar centroide con opencv
    # cv2.circle(img, (mean_x, mean_y), 5, (0, 0, 255), -1)

    list_points = []
    list_x = []
    mean_x = int(centroide[0]) + 400
    mean_y = int(centroide[2])

    for i in list_centroides:
        list_points.append(((int(i[0][0])+400, int(i[0][2])), (int(i[1][0])+400, int(i[1][2]))))
        list_x.append(int(i[0][0])+(big_size//2))
        list_x.append(int(i[1][0])+(big_size//2))
    
    val_f = True
    for points in sorted(list_points, key=lambda x: x[0][0]):
        val_f = draw_line_with_conditions(img, points[0], points[1], mean_x, val_f)

    if len(list_points) == 1:
        draw_line_with_conditions(img, list_points[0][1], list_points[0][0], mean_x, val_f)

    if mean_y-half_re_size < 0:
        img_crop = img[:re_size, :]
    elif mean_y+half_re_size > big_size:
        img_crop = img[big_size-re_size:, :]
    else:
        img_crop = img[mean_y-half_re_size:mean_y+half_re_size, :]

    if mean_x-half_re_size < 0:
        img_crop = img_crop[:, :re_size]
    elif mean_x+half_re_size > big_size:
        img_crop = img_crop[:, big_size-re_size:]
    else:
        img_crop = img_crop[:, mean_x-half_re_size:mean_x+half_re_size]

    # girar img_crop 180 grados
    # img_crop = cv2.rotate(img_crop, cv2.ROTATE_180)
    # flip
    img_crop = cv2.flip(img_crop, 0)

    print("Save gray_image", "images/shape/gray_image_" + str(name_common) + str(step_frames) + ".jpg")
    # save gray_image
    cv2.imwrite("images/shape/gray_image_" + str(name_common) + str(step_frames) + ".jpg", img_crop)
