from __future__ import print_function
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
    params.minThreshold = config.blob_detect['minThreshold']
    params.maxThreshold = config.blob_detect['maxThreshold']

    # Filter by Area.
    params.filterByArea = config.blob_detect['filterByArea']
    params.minArea = config.blob_detect['minArea']
    # params.maxArea = 100

    # Filter by Circularity
    params.filterByCircularity = config.blob_detect['filterByCircularity']
    params.minCircularity = config.blob_detect['minCircularity']

    # Filter by Convexity
    params.filterByConvexity = config.blob_detect['filterByConvexity']
    params.minConvexity = config.blob_detect['minConvexity']

    # Filter by Inertia
    params.filterByInertia = config.blob_detect['filterByInertia']
    params.minInertiaRatio =config.blob_detect['minInertiaRatio']

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
            fgMask = cv2.circle(fgMask, center_coordinates, radius, color, cv2.FILLED)  # cv2.FILLED

    fgMask = cv2.bitwise_not(fgMask)  # invert image to original
    return fgMask

def extract_fgMask_list(frame_list,background__median,background50_50):
    ##############
    # frame = cv2.medianBlur(frame, 5)
    # frame = cv2.bilateralFilter(frame, 9, 75, 75)
    # cv2.imshow('real Frame 1', frame)
    # frame = 255 - cv2.absdiff(frame,background_from_median_filter)
    # cv2.imshow('real Frame 2', frame)
    # kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    # frame = cv2.filter2D(frame, -1, kernel)
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    # frame = cv2.dilate(frame, kernel, iterations=2)  # wider

    # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # frame[:, :, 0] = cv2.equalizeHist(frame[:, :, 0])
    # frame[:, :, 1] = cv2.equalizeHist(frame[:, :, 1])
    # frame[:, :, 2] = cv2.equalizeHist(frame[:, :, 2])
    # frame = cv2.cvtColor(frame, cv2.COLOR_HSV2RGB)
    # cv2.imshow('equalizeHist', frame)

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
        backSub = cv2.createBackgroundSubtractorMOG2()
    else:
        backSub = cv2.createBackgroundSubtractorKNN(history=config.createBackground_Substraction['history'],
                                                   dist2Threshold=config.createBackground_Substraction[
                                                       'dist2Threshold'],
                                                   detectShadows=config.createBackground_Substraction['detectShadows'])
    fgMask_list = []
    i=0
    #explanation for number 102 - this is the number until which the comb fitler is efficient
    combMask_list = extract_combMask_list(frame_list)
    for frame in tqdm(frame_list):
        fgMask = backSub.apply(frame, learningRate=config.backSub_apply['learningRate'])
        # frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

        ## [apply]

        ## [display_frame_number]
        # get the frame number and write it on the current frame
        # cv2.rectangle(frame, (10, 2), (100,20), (255,255,255), -1)
        # cv2.putText(frame, str(capture.get(cv2.CAP_PROP_POS_FRAMES)), (15, 15),
        #           cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
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

        #!!!!!!!!!!!!!!!!!!!!
        # fgMask=morphological_filters(fgMask)
        # fgMask=blob_decetor(fgMask)
        #!!!!!!!!!!!!!!!!!!!!!!

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
        fgMask = morphological_filters(fgMask)
        fgMask = blob_decetor(fgMask)

        #comb filter addition first selected frames
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


def extract_combMask_list(frame_list):
    comb_mask_list = []
    i=0
    for frame in tqdm(frame_list):
        if (i==108):
            a=1
        shorts_mask = cv2.inRange(frame, config.comb_shorts['lower_bound'], config.comb_shorts['upper_bound'])
        skin_mask = cv2.inRange(frame, config.comb_skin['lower_bound'], config.comb_skin['upper_bound'])
        shirt_mask = cv2.inRange(frame, config.comb_shirt['lower_bound'], config.comb_shirt['upper_bound'])
        shoes_mask=cv2.inRange(frame,config.comb_shoes['lower_bound'], config.comb_shoes['upper_bound'])
        shoes_mask[:int(3 * shoes_mask.shape[0] / 4)] = 0
        legs_mask = cv2.inRange(frame, config.comb_legs['lower_bound'], config.comb_legs['upper_bound'])

        shirt_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, config.comb_shirt['er_struct_size'])
        shirt_mask_morph = cv2.erode(shirt_mask, shirt_kernel, iterations=config.comb_shirt['er_iterations'])
        shirt_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, config.comb_shirt['dil_struct_size'])
        shirt_mask_morph = cv2.dilate(shirt_mask_morph, shirt_kernel, iterations=config.comb_shirt['dil_iterations'])

        skin_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,config.comb_skin['er_struct_size'])
        skin_mask_morph = cv2.erode(skin_mask, skin_kernel, iterations=config.comb_skin['er_iterations'])
        skin_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, config.comb_skin['dil_struct_size'])
        skin_mask_morph = cv2.dilate(skin_mask_morph, skin_kernel, iterations=config.comb_skin['dil_iterations'])

        shorts_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, config.comb_shorts['er_struct_size'])
        shorts_mask_morph = cv2.erode(shorts_mask, shorts_kernel, iterations=config.comb_shorts['er_iterations'])
        shorts_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, config.comb_shorts['dil_struct_size'])
        shorts_mask_morph = cv2.dilate(shorts_mask_morph, shorts_kernel, iterations=config.comb_shorts['dil_iterations'])

        shoes_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, config.comb_shoes['er_struct_size'])
        shoes_mask_morph = cv2.erode(shoes_mask, shoes_kernel, iterations=config.comb_shoes['er_iterations'])
        shoes_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, config.comb_shoes['dil_struct_size'])
        shoes_mask_morph = cv2.dilate(shoes_mask_morph, shoes_kernel, iterations=config.comb_shoes['dil_iterations'])

        legs_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, config.comb_legs['er_struct_size'])
        legs_mask_morph = cv2.erode(legs_mask, legs_kernel, iterations=config.comb_legs['er_iterations'])
        legs_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, config.comb_legs['dil_struct_size'])
        legs_mask_morph = cv2.dilate(legs_mask_morph, legs_kernel, iterations=config.comb_legs['dil_iterations'])
        legs_mask_morph[:int(2 * legs_mask_morph.shape[0] / 3)] = 0

        comb_mask = cv2.bitwise_or(shirt_mask_morph, shorts_mask_morph)
        comb_mask=cv2.bitwise_or(comb_mask,shoes_mask_morph)
        comb_mask = cv2.bitwise_or(comb_mask, legs_mask_morph)
        if (i<3*len(frame_list)/8):
            comb_mask = cv2.bitwise_or(comb_mask, skin_mask_morph)
            comb_mask[:,int(frame.shape[1]/2):]=0
        elif (i>5*len(frame_list)/8):
            comb_mask[:, :int(frame.shape[1] / 2)] = 0
        comb_mask_list.append(comb_mask)
        i+=1
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
    # cv2.BackgroundSubtractorKNN.setkNNSamples(backSub,50) #How many nearest neighbours need to match.
    # cv2.BackgroundSubtractorKNN.setNSamples(backSub,1) #Sets the number of data samples in the background model.

    ## [capture]
    if (config.DEMO):
        capture = cv2.VideoCapture(config.demo_stabilized_vid_file)
    else:
        capture = cv2.VideoCapture(config.stabilized_vid_file)
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
    background__median,background50_50=median_video_improved.median_background(frame_list,config.medianSaved,config.median_background_img,config.median_background50_img)

    #seq_filter=median_video_improved.sequental_filter(frame_list)
    # frame_list_rev, fgMask_list_reversed=reversed_process(frame_list)
    #print("\nprocessing forward frames..")
    fgMask_list_forward = extract_fgMask_list(frame_list,background__median,background50_50)

    # hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    # hsv[:,:,1] = hsv[:,:,1]*fgMask
    # frame = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    # print(frame.shape)  (1080,1920,3)

    ## [show - DEBUG]
    # show the current frame and the fg masks

    # cv2.imshow('Frame', frame)
    # cv2.imshow('FG Mask', fgMask)
    # wait a few seconds
    # keyboard = cv2.waitKey(1)
    # if keyboard == 'q' or keyboard == 27:
    #    break

    # [write frame to .avi]
    print("\nwriting 'extracted.avi' and 'binary.avi'")
    img2gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
    for i in tqdm(range(len(frame_list))):
        frame = frame_list[i]
        # frame_list[i]=cv2.bitwise_and(frame_list[i],frame_list[i],mask=mask)
        # frame_list_rev[i]=cv2.bitwise_and(frame_list_rev[i],frame_list_rev[i],mask=mask)
        # frame=cv2.add(frame_list[i],frame_list_rev[i])
        fgMask = fgMask_list_forward[i]
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
