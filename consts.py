import cv2

size_centroide = 50
size_centroide_centroide = 100

configs = {
    'matlab_1': {
        'LEFT_VIDEO': '../videos/rectified/matlab_1/left_rectified.avi',
        'RIGHT_VIDEO': '../videos/rectified/matlab_1/right_rectified.avi',
        'MATRIX_Q': './config_files/matlab_1/newStereoMap.xml',
        'disparity_to_depth_map': 'disparity2depth_matrix',
        'model': "./datasets/models/z_estimation_matlab_1_keypoint_ln_model.pkl",
        'numDisparities': 68,
        'blockSize': 7, 
        'minDisparity': 5,
        'disp12MaxDiff': 33,
        'uniquenessRatio': 10,
        'preFilterCap': 33,
        'mode': cv2.StereoSGBM_MODE_HH
    },
    'opencv_1': {
        'LEFT_VIDEO': '../videos/rectified/opencv_1/left_rectified.avi',
        'RIGHT_VIDEO': '../videos/rectified/opencv_1/right_rectified.avi',
        'MATRIX_Q': './config_files/opencv_1/stereoMap.xml',
        'disparity_to_depth_map': 'disparityToDepthMap',
        'model': "./datasets/models/z_estimation_opencv_1_keypoint_ln_model.pkl",
        'numDisparities': 52,
        'blockSize': 10, 
        'minDisparity': 0,
        'disp12MaxDiff': 36,
        'uniquenessRatio': 39,
        'preFilterCap': 25,
        'mode': cv2.StereoSGBM_MODE_HH
    },
    'matlab_2': {
        'LEFT_VIDEO': '../videos/rectified/matlab_2/left_rectified.avi',
        'RIGHT_VIDEO': '../videos/rectified/matlab_2/right_rectified.avi',
        'MATRIX_Q': './config_files/laser_config/including_Y_rotation_random/iyrrStereoMap.xml',
        'disparity_to_depth_map': 'disparity2depth_matrix',
        'model': "./datasets/models/z_estimation_matlab_2_keypoint_ln_model.pkl",
        'numDisparities': 68,
        'blockSize': 7, 
        'minDisparity': 5,
        'disp12MaxDiff': 33,
        'uniquenessRatio': 10,
        'preFilterCap': 33,
        'mode': cv2.StereoSGBM_MODE_HH
    }
}