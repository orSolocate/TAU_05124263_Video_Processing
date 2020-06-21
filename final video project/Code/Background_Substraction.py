from __future__ import print_function
import cv2
import numpy as np
import argparse
import config
import logging
from tqdm import tqdm
import BS_filters
import video_handling

parser = argparse.ArgumentParser(description='KNN parser')
parser.add_argument('--algo', type=str, help='Background subtraction method (KNN, MOG2).', default='KNN')
args = parser.parse_args()


def extract_fgMask_list(frame_list,background__median,background50_50):
    if args.algo == 'MOG2':
        backSub = cv2.createBackgroundSubtractorMOG2()
    else:
        backSub = cv2.createBackgroundSubtractorKNN(history=config.createBackground_Substraction['history'],
                                                   dist2Threshold=config.createBackground_Substraction[
                                                       'dist2Threshold'],
                                                   detectShadows=config.createBackground_Substraction['detectShadows'])
    fgMask_list = []
    i=0
    combMask_list = BS_filters.extract_combMask_list(frame_list)
    for frame in tqdm(frame_list):
        fgMask = backSub.apply(frame, learningRate=config.backSub_apply['learningRate'])
        #cast to binary map
        fgMask[fgMask < 254] = 0
        #conver to [0,1] map
        fgMask = np.uint8(fgMask / 255)

        # FRAME MANIPULATION!#
        if (i==12):
            a=1
        frame[:, :, 0] = fgMask * frame[:, :, 0]
        frame[:, :, 1] = fgMask * frame[:, :, 1]
        frame[:, :, 2] = fgMask * frame[:, :, 2]
        diff = cv2.absdiff(frame, background__median)
        a = (diff[:, :, 0] <= config.mask_max_diff_from_median)
        b = (diff[:, :, 1] <= config.mask_max_diff_from_median)
        c = (diff[:, :, 2] <= config.mask_max_diff_from_median)
        a1 = np.logical_and(a, b)
        a2 = np.logical_and(a1, c)
        mask=np.ones((frame.shape[0],frame.shape[1]),dtype=np.uint8)
        mask[a2]=0
        #diff = np.mean(diff, axis=-1) averaging or using H component is not good enough
        #convert to HSV color space
        # background_hsv=cv2.cvtColor(background__median,cv2.COLOR_BGR2HSV)
        # background50_hsv=cv2.cvtColor(background50_50,cv2.COLOR_BGR2HSV)
        # frame_hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        # diff=cv2.subtract(background_hsv[:,:,0],frame_hsv[:,:,0])
        #diff = np.asarray(diff, dtype=np.uint8)

       #diff50=cv2.subtract(background50_hsv[:,:,0],frame_hsv[:,:,0])
        diff = cv2.absdiff(frame, background50_50)
        a = (diff[:, :, 0] <= config.mask_max_diff_from_median)
        b = (diff[:, :, 1] <= config.mask_max_diff_from_median)
        c = (diff[:, :, 2] <= config.mask_max_diff_from_median)
        a1 = np.logical_and(a, b)
        a2 = np.logical_and(a1, c)
        mask50 = np.ones((frame.shape[0], frame.shape[1]),dtype=np.uint8)
        mask50[a2] = 0
        #diff50 = np.asarray(diff50, dtype=np.uint8)
        #fgMask = fgMask / 255
        #fgMask[diff <= config.mask_max_diff_from_median] = 0
        fgMask[mask==0] = 0
        frame[:, :, 0] = fgMask * frame[:, :, 0]
        frame[:, :, 1] = fgMask * frame[:, :, 1]
        frame[:, :, 2] = fgMask * frame[:, :, 2]
        fgMask[mask50==0] = 0
        #fgMask[diff50 <= config.mask_max_diff_from_median] = 0
        frame[:, :, 0] = fgMask * frame[:, :, 0]
        frame[:, :, 1] = fgMask * frame[:, :, 1]
        frame[:, :, 2] = fgMask * frame[:, :, 2]

        #area_median_filter=median_video_improved.area_median_filter(frame, background50_50)
        #fgMask=cv2.bitwise_or(fgMask,area_median_filter)
        #frame[:, :, 0] = fgMask * frame[:, :, 0]
        #frame[:, :, 1] = fgMask * frame[:, :, 1]
        #frame[:, :, 2] = fgMask * frame[:, :, 2]
        #
        # area_median_filter = median_video_improved.area_median_filter(frame, background__median)
        # fgMask = cv2.bitwise_or(fgMask, area_median_filter)
        # frame[:, :, 0] = fgMask * frame[:, :, 0]
        # frame[:, :, 1] = fgMask * frame[:, :, 1]
        # frame[:, :, 2] = fgMask * frame[:, :, 2]

        #

        #         # difference = cv2.subtract(frame, background_after_median)
        #         # difference_grayscale = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
        #         # ret, mask = cv2.threshold(difference_grayscale, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        #         # fgMask[mask != 255] = 0
        fgMask = fgMask * 255
        fgMask = BS_filters.morphological_filters(fgMask)
        fgMask = BS_filters.blob_decetor(fgMask)

        #comb filter manipulation first selected frames
        if (i<config.combMask_until_this_frame):
            fgMask=combMask_list[i]
        elif (i<len(combMask_list)):
            fgMask[combMask_list[i]==255]=255
        #last manipulation
        fgMask = fgMask / 255
        frame[:, :, 0] = fgMask * frame[:, :, 0]
        frame[:, :, 1] = fgMask * frame[:, :, 1]
        frame[:, :, 2] =  fgMask * frame[:, :, 2]

        fgMask_list.append(fgMask)
        i+=1
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
    print("\nBackground_Substraction Block:")
    ## [input video]
    if (config.DEMO):
        capture = cv2.VideoCapture(config.demo_stabilized_vid_file)
    else:
        capture = cv2.VideoCapture(config.stabilized_vid_file)
    n_frames, fourcc, fps, out_size = video_handling.extract_video_params(capture)
    ## [output videos]
    out = cv2.VideoWriter(config.extracted_vid_file, fourcc, fps, out_size)
    out_bin = cv2.VideoWriter(config.binary_vid_file, fourcc, fps, out_size)

    print("\nextracting frames..")
    frame_list = []
    for i in tqdm(range(n_frames-config.BS_frame_reduction_DEBUG)):
        ret, frame = capture.read()
        if (ret == False):  # sanity check
            break
        frame_list.append(frame)

    #median filters
    background__median,background50_50=BS_filters.median_background(frame_list, config.medianSaved, config.median_background_img, config.median_background50_img)

    #sequantal filter - long runtime. unused. here for doc. proof.
    #seq_filter=median_video_improved.sequental_filter(frame_list)
    #reversed filter, unused. here for doc. proof
    # frame_list_rev, fgMask_list_reversed=reversed_process(frame_list)

    print("\nprocessing frames..")
    fgMask_list = extract_fgMask_list(frame_list,background__median,background50_50)

    # [write frame to .avi]
    print("\nwriting 'extracted.avi' and 'binary.avi'")
    for i in tqdm(range(len(frame_list))):
        #the remarks in this loop scope were used for reversed filter. kept here for doc. proof.
        frame = frame_list[i]
        # frame_list[i]=cv2.bitwise_and(frame_list[i],frame_list[i],mask=mask)
        # frame_list_rev[i]=cv2.bitwise_and(frame_list_rev[i],frame_list_rev[i],mask=mask)
        # frame=cv2.add(frame_list[i],frame_list_rev[i])
        fgMask = fgMask_list[i]
        #fgMask = combMask_list[i]
        # fgMask_list_forward[i] = cv2.bitwise_and(fgMask_list_forward[i], fgMask_list_forward[i], mask=mask)
        # fgMask_list_reversed[i] = cv2.bitwise_and(fgMask_list_reversed[i], fgMask_list_reversed[i], mask=mask)
        # fgMask = cv2.add(fgMask_list_forward[i], fgMask_list_reversed[i])
        # fgMask=cv2.bitwise_or(fgMask_list_forward[i],fgMask_list_reversed[i])
        cv2.putText(frame, "frame number : " + str(i), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)
        out.write(frame)
        fgMask = np.uint8(255 * fgMask)
        fgMask = cv2.cvtColor(fgMask, cv2.COLOR_GRAY2RGB)
        out_bin.write(fgMask)
        logging.debug('processing frame number %d', i)
    # Release video
    cv2.destroyAllWindows()
    capture.release()
    out.release()
    out_bin.release()
    return
