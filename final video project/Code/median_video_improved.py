import cv2
import numpy as np
import config
from tqdm import tqdm


def median_background(frame_list, isSaved, out_str,out_str2):
    n_frames = len(frame_list)
    frame_history_b = []
    frame_history_g = []
    frame_history_r = []
    print('process median filter')
    for i in tqdm(range(0, n_frames)):
        frame_history_b.append(frame_list[i][:, :, 2])
        frame_history_g.append(frame_list[i][:, :, 1])
        frame_history_r.append(frame_list[i][:, :, 0])
    median_frames = config.median_filter_frames_num
    if (median_frames % 2 == 0):  # even specal case
        if (median_frames >= n_frames):
            frame_history_b.append(frame_history_b[n_frames - 1])
            frame_history_g.append(frame_history_g[n_frames - 1])
            frame_history_r.append(frame_history_r[n_frames - 1])
        else:
            median_frames += 1
    frame_history_b = np.asarray(frame_history_b)
    frame_history_g = np.asarray(frame_history_g)
    frame_history_r = np.asarray(frame_history_r)
    #background all frames
    output_picture_b = np.median(frame_history_b[0:n_frames, :, :], axis=0)
    output_picture_g = np.median(frame_history_g[0:n_frames, :, :], axis=0)
    output_picture_r = np.median(frame_history_r[0:n_frames, :, :], axis=0)

    background = np.dstack((output_picture_r, output_picture_g, output_picture_b))
    background=np.asarray(background,dtype=np.uint8)

    right_picture_b = np.median(frame_history_b[0:median_frames, :, :], axis=0)
    right_picture_g = np.median(frame_history_g[0:median_frames, :, :], axis=0)
    right_picture_r = np.median(frame_history_r[0:median_frames, :, :], axis=0)
    left_picture_b = np.median(frame_history_b[len(frame_history_b)-median_frames:, :, :], axis=0)
    left_picture_g = np.median(frame_history_g[len(frame_history_g)-median_frames:, :, :], axis=0)
    left_picture_r = np.median(frame_history_r[len(frame_history_r)-median_frames:, :, :], axis=0)

    left_background=np.dstack((left_picture_r, left_picture_g, left_picture_b))
    left_background[:,int(left_background.shape[1]/2):,:]=0
    left_background=np.asarray(left_background,dtype=np.uint8)
    right_background=np.dstack((right_picture_r, right_picture_g, right_picture_b))
    right_background[:,:int(left_background.shape[1]/2),:]=0
    right_background=np.asarray(right_background,dtype=np.uint8)
    background50_50=cv2.bitwise_or(left_background,right_background)
    # gaussian_3 = cv2.GaussianBlur(background, (9,9), 10.0)
    # background = cv2.addWeighted(background, 1.5, gaussian_3, -0.5, 0, background)
    background50_50=np.asarray(background50_50,dtype=np.uint8)
    if (isSaved):
        cv2.imwrite(out_str, background)
        cv2.imwrite(out_str2,background50_50)
        cv2.destroyAllWindows()
    return background,background50_50


def sequental_filter(frame_list):
    n_frames = len(frame_list)
    frame_list_cpy=frame_list.copy()
    print('process sequental filter')
    for i in tqdm(range(0, n_frames)):
        frame_list_cpy[i] = cv2.cvtColor(frame_list_cpy[i],cv2.COLOR_BGR2HSV)
    diff=np.ones(frame_list_cpy[0].shape)
    i=0
    for frame in tqdm(frame_list_cpy):
        if (i<config.median_filter_frames_num):
            diff[frame[:,:,0]== np.any(frame_list_cpy[config.median_filter_frames_num:][:,:,0])]=0
        elif (i>config.median_filter_frames_num+40):
            diff[frame[:,:,0]== np.any(frame_list_cpy[0:config.median_filter_frames_num][:,:,0])]=0
        i+=1
    diff = np.uint8(diff*255)
    return diff


def area_median_filter(frame,background_img):
    print('process area_median_filter filter')
    frame_hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    background_hsv=cv2.cvtColor(background_img,cv2.COLOR_BGR2HSV)
    rows=frame.shape[0]
    cols=frame.shape[1]
    diff=np.ones((rows,cols))
    for i in range (config.area_filter_parameter,rows-config.area_filter_parameter):
        for j in range (config.area_filter_parameter,cols-config.area_filter_parameter):
            if(frame_hsv[i,j,0]==np.any(background_hsv[i-config.area_filter_parameter:i+config.area_filter_parameter,j-config.area_filter_parameter:j+config.area_filter_parameter,0])):
                diff[i,j]=0
    diff = np.uint8(diff*255)
    return diff