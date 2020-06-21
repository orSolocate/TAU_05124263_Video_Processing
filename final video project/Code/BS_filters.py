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
 #           if(frame_hsv[i,j,0]==np.any(background_hsv[i-config.area_filter_parameter:i+config.area_filter_parameter,j-config.area_filter_parameter:j+config.area_filter_parameter,0])):
    #             diff[i,j]=0
            diff[frame_hsv[i, j, 0] == np.any(background_hsv[i - config.area_filter_parameter:i + config.area_filter_parameter,
                                  j - config.area_filter_parameter:j + config.area_filter_parameter, 0].flatten())] = 0
    diff = np.uint8(diff*255)
    return diff



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


def extract_combMask_list(frame_list):
    comb_mask_list = []
    i=0
    for frame in tqdm(frame_list):
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
