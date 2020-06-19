from __future__ import print_function
import cv2 as cv
import cv2
import numpy as np
import argparse
import config
import logging
from tqdm import tqdm
import median_video_improved
import video_handling

parser = argparse.ArgumentParser(description='KNN parser')
parser.add_argument('--algo', type=str, help='Background subtraction method (KNN, MOG2).', default='KNN')
args = parser.parse_args()


def morphological_filters(mask):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, config.BS_erode['struct_size'])
    fgMask = cv2.erode(mask, kernel, iterations=config.BS_erode['iterations'])
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, config.BS_dilate['struct_size'])
    fgMask = cv2.dilate(fgMask, kernel, iterations=config.BS_dilate['iterations'])
    return fgMask


def blob_decetor(mask):
    #############BLOB DETECTOR- black
    # INVERT MASK
    fgMask = cv2.bitwise_not(mask)

    # # Setup SimpleBlobDetector parameters.
    params = cv2.SimpleBlobDetector_Params()

    # Change thresholds
    params.minThreshold = 0  # 10
    params.maxThreshold = 200  # 200

    # Filter by Area.
    params.filterByArea = True
    params.minArea = 0.01  # 1500
    # params.maxArea = 100

    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.1

    # Filter by Convexity
    params.filterByConvexity = True
    params.minConvexity = 0.01  # 0.87

    # Filter by Inertia
    params.filterByInertia = True
    params.minInertiaRatio = 0.01

    # Create a detector with the parameters
    ver = (cv2.__version__).split('.')
    if int(ver[0]) < 3:
        detector = cv2.SimpleBlobDetector(params)
    else:
        detector = cv2.SimpleBlobDetector_create(params)

    # Detect blobs.
    keypoints = detector.detect(fgMask)

    # Draw detected blobs as red circles.
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures
    # the size of the circle corresponds to the size of blob

    im_with_keypoints = cv2.drawKeypoints(fgMask, keypoints, np.array([]), (0, 0, 255),
                                          cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    for keyPoint in keypoints:
        x = int(keyPoint.pt[0])
        y = int(keyPoint.pt[1])
        s = keyPoint.size  # the diameter of the blob
        # Center coordinates
        center_coordinates = (x, y)
        # Radius of circle
        radius = int(s)
        # white color in BGR
        color = (255, 255, 255)

        # Line thickness of 2 px
        thickness = 1

        # Using cv2.circle() method
        # Draw a circle with blue line borders of thickness of 2 px
        if (s <= 50):
            fgMask = cv2.circle(fgMask, center_coordinates, radius, color, cv.FILLED)  # cv.FILLED

    fgMask = cv2.bitwise_not(fgMask)  # invert image to original
    return fgMask

def extract_fgMask_list(frame_list):
    ##############
    # frame = cv2.medianBlur(frame, 5)
    # frame = cv2.bilateralFilter(frame, 9, 75, 75)
    # cv.imshow('real Frame 1', frame)
    # frame = 255 - cv2.absdiff(frame,background_from_median_filter)
    # cv.imshow('real Frame 2', frame)
    # kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    # frame = cv.filter2D(frame, -1, kernel)
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    # frame = cv2.dilate(frame, kernel, iterations=2)  # wider

    # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    # frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # frame[:, :, 0] = cv.equalizeHist(frame[:, :, 0])
    # frame[:, :, 1] = cv.equalizeHist(frame[:, :, 1])
    # frame[:, :, 2] = cv.equalizeHist(frame[:, :, 2])
    # frame = cv2.cvtColor(frame, cv2.COLOR_HSV2RGB)
    # cv.imshow('equalizeHist', frame)

    # convert image from RGB to HSV
    # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    # Histogram equalisation on the V-channel
    # frame_hsv[:, :, 2] = cv2.equalizeHist(frame_hsv[:, :, 2])
    # convert image back from HSV to RGB
    # frame = cv2.cvtColor(frame_hsv, cv2.COLOR_HSV2RGB)

    # unsharp mask
    # gaussian_3 = cv2.GaussianBlur(frame, (25, 25), 10.0)
    # unsharp_image = cv2.addWeighted(frame, 1.5, gaussian_3, -0.5, 0, frame)

    ## [apply]
    # update the background model
    # learningRate = 1 the background model is completely reinitialized from the last frame.
    # learningRate = between 0 and 1 that indicates how fast the background model is learnt.
    # learningRate = -1 some automatically chosen learning rate
    # frame = frame[int(bbox[1]):int(bbox[1]+bbox[3]),int(bbox[0]):int(1.1*(bbox[0]+bbox[2])), : ]

    # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    # frame = cv2.GaussianBlur(frame, (51, 51), 0)

    # In each iteration, calculate absolute difference between current frame and reference frame
    # difference = cv2.absdiff(gray, first_gray)

    # create Background Subtractor objects
    if args.algo == 'MOG2':
        backSub = cv.createBackgroundSubtractorMOG2()
    else:
        backSub = cv.createBackgroundSubtractorKNN(history=config.createBackground_Substraction['history'],
                                                   dist2Threshold=config.createBackground_Substraction[
                                                       'dist2Threshold'],
                                                   detectShadows=config.createBackground_Substraction['detectShadows'])
    fgMask_list = []
    for frame in tqdm(frame_list):
        fgMask = backSub.apply(frame, learningRate=config.backSub_apply['learningRate'])
        # frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

        ## [apply]

        ## [display_frame_number]
        # get the frame number and write it on the current frame
        # cv.rectangle(frame, (10, 2), (100,20), (255,255,255), -1)
        # cv.putText(frame, str(capture.get(cv.CAP_PROP_POS_FRAMES)), (15, 15),
        #           cv.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
        ## [display_frame_number]

        # yonatan's add
        ###############################################################################################
        # if (iteration <= int(n_frames / 4)): # 0 to 1
        #    fgMask[:, int(1.8*1 * fgMask.shape[1]/4):int(4*fgMask.shape[1]/4)]=0

        # if (iteration <= int(2*n_frames / 4) and (iteration >= 1*n_frames/4)): #1 to 2
        #    fgMask[:, int(0 * fgMask.shape[1]/4):int(0.8*1*fgMask.shape[1]/4)]=0
        #    fgMask[:, int(1.4*2 * fgMask.shape[1]/4):int(4*fgMask.shape[1]/4)]=0

        # if (iteration <= 3*n_frames / 4) and (iteration >= 2*n_frames/4): #2 to 3
        #    fgMask[:, int(0 * fgMask.shape[1]/4):int(0.8*2*fgMask.shape[1]/4)]=0
        #    fgMask[:, int(1.4*3 * fgMask.shape[1]/4):int(4*fgMask.shape[1]/4)]=0

        # if (iteration <= 4*n_frames / 4) and (iteration >= 3*n_frames / 4): #3 to 4
        #    fgMask[:, int(0 * fgMask.shape[1]):int(0.8*3* fgMask.shape[1] / 4)]=0
        ##############################################################################################

        # kernel = np.ones((5,5),np.uint8)
        # kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))
        fgMask[fgMask < 254] = 0
        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        # fgMask = cv2.erode(fgMask, kernel, iterations=1)  # thiner

        fgMask=morphological_filters(fgMask)
        fgMask=blob_decetor(fgMask)

        # Show blobs
        # cv2.imshow("Keypoints", im_with_keypoints)
        # cv2.waitKey(0)

        #############
        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # 20,10
        # fgMask = cv2.erode(fgMask, kernel, iterations=3)  # thiner 1
        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))  # 5,5
        # fgMask = cv2.dilate(fgMask, kernel, iterations=1)  # wider 5

        # fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_OPEN, kernel)
        # fgMask[fgMask < 150] = 0

        # fgMask = cv2.bilateralFilter(fgMask, 9, 250, 250)
        # fgMask = cv2.dilate(fgMask, kernel, iterations=4)  # wider

        # fgMask = cv2.medianBlur(fgMask, 21)
        fgMask = fgMask / 255
        fgMask_list.append(fgMask)

        # FRAME MANIPULATION!#
        frame[:, :, 0] = fgMask * frame[:, :, 0]
        frame[:, :, 1] = fgMask * frame[:, :, 1]
        frame[:, :, 2] = fgMask * frame[:, :, 2]
        # frame = fgMask * frame #for grayscale images
    return fgMask_list


def extract_combMask_list(frame_list):
    comb_mask_list = []
    for frame in tqdm(frame_list):
        shorts_lower_red = np.array([0, 0, 0])
        shorts_upper_red = np.array([50, 50, 50])
        skin_lower_red = np.array([75, 85, 140])
        skin_upper_red = np.array([110, 120, 180])
        shirt_lower_red = np.array([0, 0, 40])
        shirt_upper_red = np.array([80, 80, 100])

        shorts_mask = cv2.inRange(frame, shorts_lower_red, shorts_upper_red)
        skin_mask = cv2.inRange(frame, skin_lower_red, skin_upper_red)
        shirt_mask = cv2.inRange(frame, shirt_lower_red, shirt_upper_red)

        shirt_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 10))  # 15,10
        shirt_mask_morph = cv2.erode(shirt_mask, shirt_kernel, iterations=2)  # thiner 1
        shirt_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # 5,5
        shirt_mask_morph = cv2.dilate(shirt_mask_morph, shirt_kernel, iterations=5)  # wider 5

        skin_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # 15,10
        skin_mask_morph = cv2.erode(skin_mask, skin_kernel, iterations=2)  # thiner 1
        skin_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # 5,5
        skin_mask_morph = cv2.dilate(skin_mask_morph, skin_kernel, iterations=5)  # wider 5

        shorts_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 5))  # 15,10
        shorts_mask_morph = cv2.erode(shorts_mask, shorts_kernel, iterations=2)  # thiner 1
        shorts_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # 5,5
        shorts_mask_morph = cv2.dilate(shorts_mask_morph, shorts_kernel, iterations=5)  # wider 5

        comb_mask = cv.bitwise_or(shirt_mask_morph, skin_mask_morph)
        comb_mask = cv.bitwise_or(comb_mask, shorts_mask_morph)
        comb_mask[:,int(frame.shape[1]/2):]=0
        comb_mask_list.append(comb_mask)
    return comb_mask_list


def reversed_process(frame_list):  # it didn't help - but this is the functoin...
    frame_list_rev = frame_list.copy()
    frame_list_rev.reverse()
    print("\nprocessing reversed frames..")
    fgMask_list_reversed = extract_fgMask_list(frame_list_rev)
    frame_list_rev.reverse()
    fgMask_list_reversed.reverse()
    return frame_list_rev, fgMask_list_reversed


def Background_Substraction():
    ## [create]
    print("\nBackground_Substraction:")

    ## [create]
    # cv.BackgroundSubtractorKNN.setkNNSamples(backSub,50) #How many nearest neighbours need to match.
    # cv.BackgroundSubtractorKNN.setNSamples(backSub,1) #Sets the number of data samples in the background model.

    ## [capture]
    if (config.DEMO):
        capture = cv.VideoCapture(config.demo_stabilized_vid_file)
    else:
        capture = cv.VideoCapture(config.stabilized_vid_file)
    n_frames, fourcc, fps, out_size = video_handling.extract_video_params(capture)
    # Set up output video
    out = cv2.VideoWriter(config.extracted_vid_file, fourcc, fps, out_size)
    # Set up output video
    out_bin = cv2.VideoWriter(config.binary_vid_file, fourcc, fps, out_size)

    background_from_median_filter = cv2.imread('background_from_median_filter.jpeg')
    # background = cv2.imread(config.in_background_file)

    print("\nextracting frames..")
    frame_list = []
    for i in tqdm(range(n_frames-config.BS_frame_reduction_DEBUG)):
        ret, frame = capture.read()
        if (ret == False):  # sanity check
            break
        frame_list.append(frame)
        # if iteration==0:
        #   first_frame = frame

    #median try - Or
    background_after_median=median_video_improved.median_background(frame_list,config.medianSaved,config.median_background_img)


    # frame_list_rev, fgMask_list_reversed=reversed_process(frame_list)
    print("\nprocessing forward frames..")
    fgMask_list_forward = extract_fgMask_list(frame_list)

    # hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    # hsv[:,:,1] = hsv[:,:,1]*fgMask
    # frame = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    # print(frame.shape)  (1080,1920,3)

    ## [show - DEBUG]
    # show the current frame and the fg masks

    # cv.imshow('Frame', frame)
    # cv.imshow('FG Mask', fgMask)
    # wait a few seconds
    # keyboard = cv.waitKey(1)
    # if keyboard == 'q' or keyboard == 27:
    #    break

    # [write frame to .avi]
    print("\nwriting 'extracted.avi' and 'binary.avi'")
    img2gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    ret, mask = cv.threshold(img2gray, 10, 255, cv.THRESH_BINARY)
    for i in tqdm(range(len(frame_list))):
        frame = frame_list[i]
        # frame_list[i]=cv.bitwise_and(frame_list[i],frame_list[i],mask=mask)
        # frame_list_rev[i]=cv.bitwise_and(frame_list_rev[i],frame_list_rev[i],mask=mask)
        # frame=cv.add(frame_list[i],frame_list_rev[i])
        fgMask = fgMask_list_forward[i]
        # fgMask_list_forward[i] = cv.bitwise_and(fgMask_list_forward[i], fgMask_list_forward[i], mask=mask)
        # fgMask_list_reversed[i] = cv.bitwise_and(fgMask_list_reversed[i], fgMask_list_reversed[i], mask=mask)
        # fgMask = cv.add(fgMask_list_forward[i], fgMask_list_reversed[i])
        # fgMask=cv.bitwise_or(fgMask_list_forward[i],fgMask_list_reversed[i])
        cv.putText(frame, "frame number : " + str(i), (100, 50), cv.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)
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
