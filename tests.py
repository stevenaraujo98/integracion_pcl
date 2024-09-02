import numpy as np
from ultralytics import YOLO
import json

# Load a model
model = YOLO("./models/detect-shape_v5.pt")

def calcular_angulo_con_eje_y(normal_plano):
  if len(normal_plano) != 3:
    return -1
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
    if len(vector) != 3:
        return -1
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
    return res[0].names[res[0].probs.top1], str(res[0].probs.top1conf.item() * 100)

def get_structure_data(kps, character, list_tronco_normal, list_head_normal, avg_normal, avg_normal_head, list_centroides, list_union_centroids, centroide, head_centroid, list_is_centroid_to_nariz, list_heights):
    res = {}
    res["persons"] = {}

    for i, person in enumerate(kps):
      list_head = np.array(person)[[0, 3, 4], :].tolist()
      list_tronco = np.array(person)[[5, 6, 11, 12], :].tolist()
      res["persons"][i] = {}
      res["persons"][i]["points"] = person.tolist()
      res["persons"][i]["centroid"] = list_centroides[i].tolist()
      res["persons"][i]["points_tronco"] = list_tronco
      res["persons"][i]["tronco_normal"] = list_tronco_normal[i].tolist()
      res["persons"][i]["angle_tronco"] = calcular_angulo_con_eje_y(list_tronco_normal[i])
      res["persons"][i]["is_centroid_to_nariz"] = list_is_centroid_to_nariz[i]
      res["persons"][i]["points_head"] = list_head
      res["persons"][i]["head_normal"] = list_head_normal[i].tolist()
      res["persons"][i]["angle_head"] = calcular_angulo_con_eje_y(list_head_normal[i])
      res["persons"][i]["height"] = list_heights[i]
    
    res["count"] = i+1
    res["character"] = character
    res["centroid"] = centroide.tolist()
    res["avg_normal"] = avg_normal.tolist()
    res["angle_avg_normal"] = calcular_angulo_con_eje_y(avg_normal)
    res["centroid_head"] = head_centroid.tolist()
    res["avg_normal_head"] = avg_normal_head.tolist()
    res["angle_avg_normal_head"] = calcular_angulo_con_eje_y(avg_normal_head)

    new_list_union_centroids = []
    for i in list_union_centroids:
      list_tmp = []
      for j in i:
        list_tmp.append(j.tolist())
      new_list_union_centroids.append(list_tmp)
    res["union_centroids"] = new_list_union_centroids
    
    print(json.dumps(res))
