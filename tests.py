import numpy as np
from ultralytics import YOLO

# Load a model
model = YOLO("./models/detect-shape_v4.pt")

def calcular_angulo_con_eje_y(normal_plano):
  # Vector del eje X
  eje_x = np.array([1, 0, 0])
  
  # Calcular el producto punto
  producto_punto = np.dot(normal_plano, eje_x)
  
  # Calcular las magnitudes de los vectores
  magnitud_normal_plano = np.linalg.norm(normal_plano)
  magnitud_eje_y = np.linalg.norm(eje_x)
  
  # Calcular el coseno del ángulo
  cos_theta = producto_punto / (magnitud_normal_plano * magnitud_eje_y)
  
  # Calcular el ángulo en radianes y luego convertir a grados
  angulo_radianes = np.arccos(cos_theta)
  angulo_grados = np.degrees(angulo_radianes)
  
  return angulo_grados


def get_angulo_with_x(vector):
    # Ejemplo de uso
    angulo = calcular_angulo_con_eje_y(vector)

    print("Ángulo con el eje Y:", angulo)  # Imprime el ángulo en grados

    # Extraer las componentes X y Z del vector
    componente_x = vector[0]
    componente_z = vector[2]

    print("Componente X:", componente_x)  # Imprime la componente X
    print("Componente Z:", componente_z)  # Imprime la componente Z

def get_character(image):
    res = model(image)
    print("Letra detectada: " +  res[0].names[res[0].probs.top1] + " confianza: " + str(res[0].probs.top1conf.item() * 100))
