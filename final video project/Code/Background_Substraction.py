from __future__ import print_function
import cv2
import numpy as np
import argparse
import config
from tqdm import tqdm
import BS_filters
import video_handling
import time


parser = argparse.ArgumentParser(description='KNN parser')
parser.add_argument('--algo', type=str, help='Background subtraction method (KNN, MOG2).', default='KNN')
args = parser.parse_args()


def extract_fgMask_list(frame_list, background__median, background50_50):
    if args.algo == 'MOG2':
        backSub = cv2.createBackgroundSubtractorMOG2()
    else:
        backSub = cv2.createBackgroundSubtractorKNN(history=config.createBackground_Substraction['history'],
                                                    dist2Threshold=config.createBackground_Substraction[
                                                        'dist2Threshold'],
                                                    detectShadows=config.createBackground_Substraction['detectShadows'])
    fgMask_list = []
    i = 0
    combMask_list = BS_filters.extract_combMask_list(frame_list)
    for frame in tqdm(frame_list):
        fgMask = backSub.apply(frame, learningRate=config.backSub_apply['learningRate'])
        # cast to binary map
        fgMask[fgMask < 254] = 0
        # conver to [0,1] map
        fgMask = np.uint8(fgMask / 255)

        # FRAME MANIPULATION!#
        diff = cv2.absdiff(frame, background__median)
        a = (diff[:, :, 0] <= config.mask_max_diff_from_median)
        b = (diff[:, :, 1] <= config.mask_max_diff_from_median)
        c = (diff[:, :, 2] <= config.mask_max_diff_from_median)
        a1 = np.logical_and(a, b)
        a2 = np.logical_and(a1, c)
        mask = np.ones((frame.shape[0], frame.shape[1]), dtype=np.uint8)
        mask[a2] = 0
        diff = cv2.absdiff(frame, background50_50)
        a = (diff[:, :, 0] <= config.mask_max_diff_from_median)
        b = (diff[:, :, 1] <= config.mask_max_diff_from_median)
        c = (diff[:, :, 2] <= config.mask_max_diff_from_median)
        a1 = np.logical_and(a, b)
        a2 = np.logical_and(a1, c)
        mask50 = np.ones((frame.shape[0], frame.shape[1]), dtype=np.uint8)
        mask50[a2] = 0
        fgMask[mask == 0] = 0
        fgMask[mask50 == 0] = 0
        fgMask = fgMask * 255
        fgMask = BS_filters.morphological_filters(fgMask)
        fgMask = BS_filters.blob_decetor(fgMask)

        # comb filter manipulation first selected frames
        if (i < config.combMask_until_this_frame):
            fgMask = combMask_list[i]
        elif (i < len(combMask_list)):
            fgMask[combMask_list[i] == 255] = 255
        # last manipulation
        fgMask = fgMask / 255
        frame[:, :, 0] = fgMask * frame[:, :, 0]
        frame[:, :, 1] = fgMask * frame[:, :, 1]
        frame[:, :, 2] = fgMask * frame[:, :, 2]

        fgMask_list.append(fgMask)
        i += 1
    return fgMask_list


def reversed_process(frame_list):  # unused function. here for documentation proof.
    frame_list_rev = frame_list.copy()
    frame_list_rev.reverse()
    print("\nprocessing reversed frames..")
    fgMask_list_reversed = extract_fgMask_list(frame_list_rev)
    frame_list_rev.reverse()
    fgMask_list_reversed.reverse()
    return frame_list_rev, fgMask_list_reversed


def Background_Substraction():
    print("\nBackground_Substraction Block:\n")
    ## [input video]
    if (config.DEMO):
        capture = cv2.VideoCapture(config.demo_stabilized_vid_file)
    else:
        capture = cv2.VideoCapture(config.stabilized_vid_file)
    n_frames, fourcc, fps, out_size = video_handling.extract_video_params(capture)
    ## [output videos]
    out = cv2.VideoWriter(config.extracted_vid_file, fourcc, fps, out_size)
    out_bin = cv2.VideoWriter(config.binary_vid_file, fourcc, fps, out_size)

    print("extracting frames..")
    frame_list = []
    time.sleep(0.3)
    for i in tqdm(range(n_frames - config.BS_frame_reduction_DEBUG)):
        ret, frame = capture.read()
        if (ret == False):  # sanity check
            break
        frame_list.append(frame)

    # median filters
    time.sleep(0.3)
    background__median, background50_50 = BS_filters.median_background(frame_list, config.medianSaved,
                                                                       config.median_background_img,
                                                                       config.median_background50_img)

    print("processing frames..")
    time.sleep(0.3)
    fgMask_list = extract_fgMask_list(frame_list, background__median, background50_50)

    # [write frame to .avi]
    print("writing 'extracted.avi' and 'binary.avi'")
    time.sleep(0.3)
    for i in tqdm(range(len(frame_list))):
        frame = frame_list[i]
        fgMask = fgMask_list[i]
        cv2.putText(frame, "frame number : " + str(i), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)
        out.write(frame)
        fgMask = np.uint8(255 * fgMask)
        fgMask = cv2.cvtColor(fgMask, cv2.COLOR_GRAY2RGB)
        out_bin.write(fgMask)
    # Release video
    cv2.destroyAllWindows()
    capture.release()
    out.release()
    out_bin.release()
    return
