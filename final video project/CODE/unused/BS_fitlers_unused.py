import cv2
import numpy as np
import config
from tqdm import tqdm


def sequental_filter(frame_list):
    n_frames = len(frame_list)
    frame_list_cpy = frame_list.copy()
    print('process sequental filter')
    for i in tqdm(range(0, n_frames)):
        frame_list_cpy[i] = cv2.cvtColor(frame_list_cpy[i], cv2.COLOR_BGR2HSV)
    diff = np.ones(frame_list_cpy[0].shape)
    i = 0
    for frame in tqdm(frame_list_cpy):
        if (i < config.median_filter_frames_num):
            diff[frame[:, :, 0] == np.any(frame_list_cpy[config.median_filter_frames_num:][:, :, 0])] = 0
        elif (i > config.median_filter_frames_num + 40):
            diff[frame[:, :, 0] == np.any(frame_list_cpy[0:config.median_filter_frames_num][:, :, 0])] = 0
        i += 1
    diff = np.uint8(diff * 255)
    return diff


def area_median_filter(frame, background_img):
    print('process area_median_filter filter')
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    background_hsv = cv2.cvtColor(background_img, cv2.COLOR_BGR2HSV)
    rows = frame.shape[0]
    cols = frame.shape[1]
    diff = np.ones((rows, cols))
    for i in range(config.area_filter_parameter, rows - config.area_filter_parameter):
        for j in range(config.area_filter_parameter, cols - config.area_filter_parameter):
            #           if(frame_hsv[i,j,0]==np.any(background_hsv[i-config.area_filter_parameter:i+config.area_filter_parameter,j-config.area_filter_parameter:j+config.area_filter_parameter,0])):
            #             diff[i,j]=0
            diff[frame_hsv[i, j, 0] == np.any(
                background_hsv[i - config.area_filter_parameter:i + config.area_filter_parameter,
                j - config.area_filter_parameter:j + config.area_filter_parameter, 0].flatten())] = 0
    diff = np.uint8(diff * 255)
    return diff
