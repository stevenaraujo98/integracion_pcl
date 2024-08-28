import numpy as np
import cv2
from tests import get_character

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

    start, end = (0, 0), (1, 0)
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
    else:
        print("Error en la dirección de la línea")
    
    # Calcular la pendiente y el intercepto
    # y = m x + b
    m = (end[1] - start[1]) / (end[0] - start[0])
    b = start[1] - m * start[0]
    # print("m", m, "b", b)
    
    y_res = int(m * x_2 + b)

    cv2.line(image, start, (x_2, y_res), (0, 0, 0), 2)

    return val_f


def get_img_shape_meet_prev_sort(list_centroides_sorted, puntos, name_common, step_frames, centroide, list_pos_extremo):
    big_size = 800
    re_size = 640
    half_re_size = re_size//2
    # generar la imagen hasta el limite maximo
    img = np.ones((big_size, big_size, 3), dtype=np.uint8) * 255

    list_points = []
    mean_x = int(centroide[0]) + 400
    mean_y = int(centroide[1])
    list_inf = []

    for simplex in list_centroides_sorted:
        pos0 = simplex[0]
        pos1 = simplex[1]
        p1, p2 = puntos[simplex]
        list_points.append(((int(p1[0])+400, int(p1[1])), (int(p2[0])+400, int(p2[1]))))

        if pos0 in list_pos_extremo[0] or pos1 in list_pos_extremo[0]:
            if pos0 in list_pos_extremo[0] and not pos1 in list_pos_extremo[0]:
                p1 = puntos[pos1]
                p2 = puntos[pos0]
            elif pos1 in list_pos_extremo[0] and not pos0 in list_pos_extremo[0]:
                p1 = puntos[pos0]
                p2 = puntos[pos1]
            list_inf.append(((int(p1[0])+400, int(p1[1])), (int(p2[0])+400, int(p2[1]))))

    # unir los puntos de list_points en la imagen img
    for points in list_points:
        cv2.line(img, points[0], points[1], (0, 0, 0), 2)

    for points in sorted(list_inf, key=lambda x: x[1][0]):
        x1, y1 = points[0]
        x2, y2 = points[1]
        m = (y2 - y1) / (x2 - x1)

        if x1 - x2 != 0 and y1 - y2 != 0:
            arrow_left =  x1 > x2
            if arrow_left:
                x = 0
                b = int(y1 - m * x1)
                cv2.line(img, points[0], [x, b], (0, 0, 0), 2)
                print("arrow_left", x, b)
            else:
                x = big_size-1
                b = int(y2 - m * x2)
                y = int(m * x + b)
                cv2.line(img, points[0], [x, y], (0, 0, 0), 2)
                print("arrow_right", x, y)
        elif y1 - y2 == 0 and x1 - x2 != 0:
            arrow_left =  x1 > x2
            if arrow_left:
                cv2.line(img, points[0], [0, y1], (0, 0, 0), 2)
            else:
                cv2.line(img, points[0], [big_size-1, y2], (0, 0, 0), 2)
        elif x1 - x2 == 0 and y1 - y2 != 0:
            if y1 > y2:
                cv2.line(img, points[0], [x1, 0], (0, 0, 0), 2)
            else:
                cv2.line(img, points[0], [x2, big_size-1], (0, 0, 0), 2)

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

    img_crop = cv2.flip(img_crop, 0)
    img_res = cv2.erode(img_crop, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=15) # _thick
    img_res_2 = cv2.bitwise_not(img_res) # _not_thick

    character, confianza = get_character(img_res_2)
    print("Save gray_image", "images/shape/gray_image_" + str(name_common) + str(step_frames) + ".jpg")
    # save gray_image
    cv2.imwrite("images/shape/gray_image_" + str(name_common) + str(step_frames) + ".jpg", img_res_2)
    return character, confianza