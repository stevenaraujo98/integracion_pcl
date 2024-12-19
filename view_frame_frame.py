import cv2
import numpy as np
import time

name_common = "11_04_35_17_10_2024_VID" # [477, 518, 1005, 1055, 1124]
# name_common = "10_56_20_17_10_2024_VID" # [481, 517, 584, 781, 823, 910, 985, 1059, 1140, 1217, 1250, 1423, 1506, 1611, 1828, 1856, 1903, 2079, 2138, 2782, 2881, 2945, 2994, 3116, 3161, 3260, 3469, 3777, 3880, 3946]
path_img_L = "./datasets/190824/grupos/" + name_common + "_LEFT.avi"

# video = cv2.VideoCapture(path_img_L)

# num_frame = 0
# list_frame_obj = []
# while True:
#     ret, frame = video.read()
#     num_frame += 1

#     if not ret:
#         break

#     cv2.imshow("Frame", frame)
    
#     if cv2.waitKey(1) & 0xFF == ord('s'):
#         print("Saving frame", num_frame)
#         list_frame_obj.append(num_frame)
#     elif cv2.waitKey(1) & 0xFF == ord('q'):
#         break

#     #espere medio mili segundo
#     time.sleep(0.005)

# cv2.destroyAllWindows()
# print("List of frames", list_frame_obj)


name_common = "image_10_56_20_17_10_2024_VID3116"
img = cv2.imread("./images/kp/" + name_common + ".jpg")

cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
