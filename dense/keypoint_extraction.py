import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

from ultralytics import YOLO

from ultralytics.utils.plotting import Annotator

# torch.cuda.set_device(0)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------------------------------- KEYPOINTS EXTRACTION -------------------------------------------------------

# Load a model de models
# pose = YOLO('..models.yolov8n-pose.pt').to(device=device)  # load an official model
# segmentation = YOLO('..models.yolov8n-seg.pt').to(device=device)  # load an official model
pose = YOLO('../models/yolov8n-pose.pt')
segmentation = YOLO('../models/yolov8n-seg.pt')


# Extract results
def get_keypoints(source):
    results = pose(source=source, show=False, save = True, conf=0.8, classes=[0]) 
    keypoints = np.array(results[0].keypoints.xy.cpu())
    return keypoints

def get_roi(source):
    results = pose(source=source, show=False, save = False, conf=0.8, classes=[0])
    roi = np.array(results[0].boxes.xyxy.cpu())
    return roi

def get_segmentation(source):
    results = segmentation(source=source, show=False, save = False, conf=0.8)[0] 
    seg = results.masks.xy
    return seg

def apply_roi_mask(image, roi):
    mask = np.zeros(image.shape[:2], dtype=np.uint8) 

    # Inicializa la máscara como una copia de la máscara original (normalmente toda en ceros)
    for coor in roi:
        mask[int(coor[1]):int(coor[3]), int(coor[0]):int(coor[2])] = 1  # Pone en 1 los pixeles dentro de los cuadrados definidos

    # Aplica la máscara a la imagen
    masked_image = cv2.bitwise_and(image, image, mask=mask.astype(np.uint8) * 255)

    return masked_image

def apply_keypoints_mask(image, keypoints):
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    # Inicializa la máscara como una copia de la máscara original (normalmente toda en ceros)
    for person in keypoints:
        for kp in person:
            y, x = int(kp[1]), int(kp[0])
            # Verificar si las coordenadas están dentro de los límites de la imagen
            if 0 <= y - 1 < image.shape[0] and 0 <= x - 1 < image.shape[1]:
                mask[y - 1, x - 1] = 1  # Pone en 1 los pixeles dentro de los cuadrados definidos
        # print(person)
        # kp = person[16]
        # mask[int(kp[1]) - 1, int(kp[0]) - 1] = 1
    # Aplica la máscara a la imagen
    masked_image = cv2.bitwise_and(image, image, mask=mask.astype(np.uint8) * 255)
    return masked_image





def apply_seg_mask(image, segmentation):
    mask = np.zeros(image.shape[:2], dtype=np.uint8)  # Inicializa la máscara en ceros

    # Itera sobre cada segmento en la segmentación
    for seg in segmentation:
        # Convierte las coordenadas del segmento a un formato adecuado para fillPoly
        seg = np.array(seg, dtype=np.int32)
        # Rellena el área dentro del contorno
        cv2.fillPoly(mask, [seg], 1)

    # Aplica la máscara a la imagen
    masked_image = cv2.bitwise_and(image, image, mask=mask * 255)

    return masked_image

def apply_seg_individual_mask(image, segmentation):
    mask = np.zeros(image.shape[:2], dtype=np.uint8)  # Inicializa la máscara en ceros
    # Convierte las coordenadas del segmento a un formato adecuado para fillPoly
    segmentation = np.array(segmentation, dtype=np.int32)
    # Rellena el área dentro del contorno
    cv2.fillPoly(mask, [segmentation], 1)

    # Aplica la máscara a la imagen
    masked_image = cv2.bitwise_and(image, image, mask=mask * 255)

    return masked_image