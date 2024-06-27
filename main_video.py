import cv2
import numpy as np
import matplotlib.pyplot as plt
from space_3d import show_centroid_and_normal, calcular_centroide, show_each_point_of_person, show_connection_points
from character_meet import get_img_shape_meet

lista_colores = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
list_colors = [(255,0,255), (0, 255, 255), (255, 0, 0), (0, 0, 0), (255, 255, 0), (205, 92, 92), (255, 0, 255), (0, 128, 128), (128, 0, 0), (128, 128, 0), (128, 128, 128)]
# Parametos intrinsecos de la camara
baseline = 58
f_px = 3098.3472392388794
# f_px = (1430.012149825778 + 1430.9237520924735 + 1434.1823778243315 + 1435.2411841383973) / 4 
# center = ((929.8117736131715 + 936.7865692255547) / 2,
#            (506.4104424162777 + 520.0461674300153) / 2)
center = (777.6339950561523,539.533634185791)



figure = None
def setup_plot():
    global figure
    if figure is not None:
        return figure

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_ylim(-1500, 1500)
    ax.set_xlim(-1000, 1000)
    ax.set_zlim(-2000, 2000)
    figure = fig, ax
    return fig, ax

def plot_3d(x, y, z, ax, color, s=None, marker="o", label=None):
    if (s):
        ax.scatter(x, y, z, color=color, marker=marker, s=s)
    else:
        ax.scatter(x, y, z, color=color, marker=marker)
    
    if label:
        ax.text(x, y+25, z, label, color=color)

def clean_plot(ax):
    ax.cla()
    ax.set_ylim(-1000, 2000)
    ax.set_xlim(-3000, 3000)
    ax.set_zlim(0, 10000)

def live_plot_3d(left_kpts, right_kpts, baseline, f_px, center):
    fig, ax = setup_plot()
    clean_plot(ax)
    list_points_persons = []
    list_color_to_paint = []
    list_centroides = []

    # Agregar a una lista de colores para pintar los puntos de cada persona en caso de ser mas de len(lista_colores)
    for i in range(len(left_kpts)):
        indice_color = i % len(lista_colores)
        list_color_to_paint.append(lista_colores[indice_color])

    show_each_point_of_person(left_kpts, right_kpts, baseline, f_px, center, list_color_to_paint, ax, plot_3d, list_points_persons)
    show_centroid_and_normal(list_points_persons, list_color_to_paint, ax, list_centroides, plot_3d)

    if len(list_centroides) > 1:
        # Ilustrar el centroide de los centroides (centroide del grupo)
        centroide = calcular_centroide(list_centroides)
        # plot_3d(centroide[0], centroide[1], centroide[2], ax, "black", s=800, marker='o', label="Cg")

    # Conectar cada uno de los ceintroides
    show_connection_points(list_centroides, ax)

    # get_img_shape_meet(list_centroides)
    
    plt.draw()
    plt.pause(0.0001)
    return list_points_persons

def point_cloud(left_kpts, right_kpts, baseline, f_px, center):
    return live_plot_3d(left_kpts, right_kpts, baseline, f_px, center)

def getStereoRectifier(calib_file):
    """Build rectifier from stereo map file
    Parameters:
        calib_file (str): file name of the stereo map file generated with calibration procedure
    Returns:
        (np.ndarray, np.ndarray)->(np.ndarray,np.ndarray) rectify function takes 2 unrectified images and returns those images calibrated
    """

    # Camera parameters to undistort and rectify images
    cv_file = cv2.FileStorage()
    cv_file.open(calib_file, cv2.FileStorage_READ)

    stereoMapL_x = cv_file.getNode('stereoMapL_x').mat()
    stereoMapL_y = cv_file.getNode('stereoMapL_y').mat()
    stereoMapR_x = cv_file.getNode('stereoMapR_x').mat()
    stereoMapR_y = cv_file.getNode('stereoMapR_y').mat()

    def undistortRectify(frameL, frameR):

        # Undistort and rectify images
        undistortedL = cv2.remap(frameL, stereoMapL_x, stereoMapL_y,
                                 cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
        undistortedR = cv2.remap(frameR, stereoMapR_x, stereoMapR_y,
                                 cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)

        return undistortedL, undistortedR

    return undistortRectify


def graph_circles(array_pts, w, h, frame):
    num_person = 0
    for person in array_pts:
        for pt in person:
            pos_x = int(pt[0])
            pos_y = int(pt[1])
            if (pos_x != 0 and pos_y != 0):
                # print(pos_x, pos_y)
                cv2.circle(frame, (pos_x, pos_y), 3, list_colors[num_person], 3)
        num_person+=1


# YOLO -----------------------------------------------------------------------------------------------------------------
capL=cv2.VideoCapture('./predicts/YOLO/newcalibration/15_58_03_05_04_2024_VIDEO_LEFT.avi')
path_file_left = './predicts/YOLO/newcalibration/15_58_03_05_04_2024_VIDEO_LEFT/15_58_03_05_04_2024_VIDEO_LEFT_'
capR=cv2.VideoCapture('./predicts/YOLO/newcalibration/15_58_03_05_04_2024_VIDEO_RIGHT.avi')
path_file_right = './predicts/YOLO/newcalibration/15_58_03_05_04_2024_VIDEO_RIGHT/15_58_03_05_04_2024_VIDEO_RIGHT_'


# capL=cv2.VideoCapture('./predicts/YOLO/integradora/16_35_42_26_02_2024_VID_LEFT.avi')
# path_file_left = './predicts/YOLO/integradora/16_35_42_26_02_2024_VID_LEFT/16_35_42_26_02_2024_VID_LEFT_'
# capR=cv2.VideoCapture('./predicts/YOLO/integradora/16_35_42_26_02_2024_VID_RIGHT.avi')
# path_file_right = './predicts/YOLO/integradora/16_35_42_26_02_2024_VID_RIGHT/16_35_42_26_02_2024_VID_RIGHT_'


# OpenPose -------------------------------------------------------------------------------------------------------------
# capL=cv2.VideoCapture('./predicts/OP/newcalibration/15_58_03_05_04_2024_VIDEO_LEFT.avi')
# path_file_left = './predicts/OP/newcalibration/15_58_03_05_04_2024_VIDEO_LEFT/frame_'
# capR=cv2.VideoCapture('./predicts/OP/newcalibration/15_58_03_05_04_2024_VIDEO_RIGHT.avi')
# path_file_right = './predicts/OP/newcalibration/15_58_03_05_04_2024_VIDEO_RIGHT/frame_'

"""
# LightGlue ------------------------------------------------------------------------------------------------------------
capL=cv2.VideoCapture('./predicts/LightGlue/16_35_42_26_02_2024_VID_LEFT_calibrated.avi')
path_file_left = './predicts/LightGlue/16_35_42_26_02_2024_VID_LEFT/16_35_42_26_02_2024_VID_LEFT_'
capR=cv2.VideoCapture('./predicts/LightGlue/16_35_42_26_02_2024_VID_RIGHT_calibrated.avi')
path_file_right = './predicts/LightGlue/16_35_42_26_02_2024_VID_RIGHT/16_35_42_26_02_2024_VID_RIGHT_'
"""

frame_num = 1
start_frame = 705 # C #269 # L # 1000 # I #  256
# Video
# start_frame = 1000

while(capR.isOpened() and capL.isOpened()):
    if frame_num < start_frame:
        ret,frameL = capL.read()
        retR,frameR = capR.read()
        frame_num += 1
        continue

    ret,frameL = capL.read()
    retR,frameR = capR.read()

    h = frameR.shape[0]
    w = frameR.shape[1]

    if(not ret or not retR):
        print("Failed to read frames")
        break
    
    cv2.imshow('LEFT',frameL)
    cv2.imshow('RIGHT',frameR)

    keypointsL = np.array(getKeypoints(path_file_left + str(frame_num) + '.txt'))
    keypointsR_sorted = np.array(getKeypoints(path_file_right + str(frame_num) + '.txt'))

    # keypointsL_copy = keypointsL.copy()
    # keypointsL_copy[:, :, 0] += 8
    key = cv2.waitKey(0)
    if key == ord('q'):
        # Close
        break

    if (len(keypointsR_sorted) == 0 or len(keypointsL) == 0):
        print("No keypoints found in frame", frame_num, len(keypointsL), len(keypointsR_sorted))
        frame_num += 1
        continue

    # print(keypointsL[:, [5, 6, 11, 12], :])
    # print(keypointsR_sorted[:, [5, 6, 11, 12], :])

    # YOLO
    keypointsL_filtered = keypointsL[:, [0, 3, 4, 5, 6, 11, 12], :]
    keypointsR_filtered = keypointsR_sorted[:, [0, 3, 4, 5, 6, 11, 12], :]

    # OpenPose
    # keypointsL_filtered = keypointsL[:, [0, 2, 5, 9, 12], :]
    # keypointsR_filtered = keypointsR_sorted[:, [0, 2, 5, 9, 12], :]
    lists_points_3d = point_cloud(keypointsL_filtered, keypointsR_filtered, baseline, f_px, center)
    frame_num += 1
    print("Frame", frame_num)

capL.release()
capR.release()
cv2.destroyAllWindows()
