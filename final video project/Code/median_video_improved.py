import cv2
import numpy as np
import config
from tqdm import tqdm


def median_background(frame_list, isSaved, out_str):
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
    output_picture_b = np.median(frame_history_b[0:median_frames, :, :], axis=0)
    output_picture_g = np.median(frame_history_g[0:median_frames, :, :], axis=0)
    output_picture_r = np.median(frame_history_r[0:median_frames, :, :], axis=0)
    background = np.dstack((output_picture_r, output_picture_g, output_picture_b))
    # gaussian_3 = cv2.GaussianBlur(background, (9,9), 10.0)
    # background = cv2.addWeighted(background, 1.5, gaussian_3, -0.5, 0, background)
    background=np.asarray(background,dtype=np.uint8)
    if (isSaved):
        cv2.imwrite(out_str, background)
        cv2.destroyAllWindows()
    return background
