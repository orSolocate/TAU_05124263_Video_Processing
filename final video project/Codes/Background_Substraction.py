from __future__ import print_function
import cv2 as cv
import cv2
import numpy as np
import argparse
import sys


parser = argparse.ArgumentParser(description='This program shows how to use background subtraction methods provided by \
                                              OpenCV. You can process both videos and images.')
parser.add_argument('--input', type=str, help='Path to a video or a sequence of image.', default='Stabilized_Example_INPUT.avi')
parser.add_argument('--algo', type=str, help='Background subtraction method (KNN, MOG2).', default='KNN')
args = parser.parse_args()

## [create]
#create Background Subtractor objects
if args.algo == 'MOG2':
    backSub = cv.createBackgroundSubtractorMOG2()
else:
    backSub = cv.createBackgroundSubtractorKNN(history = 120, dist2Threshold = 400.0, detectShadows = True)
## [create]


## [capture]
capture = cv.VideoCapture('Stabilized_Example_INPUT.avi')
#capture = cv.VideoCapture('video_out.avi')
n_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
if not capture.isOpened:
    print('Unable to open: ' + args.input)
    exit(0)
## [capture]

# Get width and height of video stream
w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Get frames per second (fps)
fps = capture.get(cv2.CAP_PROP_FPS)

# Define the codec for output video
fourcc = cv2.VideoWriter_fourcc(*'MJPG')

# Set up output video
out = cv2.VideoWriter('Backgroud_Substraction_out.avi', fourcc, fps, (w, h))
# Set up output video
out_bin = cv2.VideoWriter('binary.avi', fourcc, fps, (w, h))


iteration= 1
background = cv2.imread('background.jpeg')

#########################################################
# Start timer
timer = cv2.getTickCount()

    # Set up tracker.
    # Instead of MIL, you can also use

tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'CSRT']
tracker_type = tracker_types[2]

if tracker_type == 'BOOSTING':
    tracker = cv2.TrackerBoosting_create()
if tracker_type == 'MIL':
    tracker = cv2.TrackerMIL_create()
if tracker_type == 'KCF':
    tracker = cv2.TrackerKCF_create()
if tracker_type == 'TLD':
    tracker = cv2.TrackerTLD_create()
if tracker_type == 'MEDIANFLOW':
    tracker = cv2.TrackerMedianFlow_create()
if tracker_type == 'GOTURN':
    tracker = cv2.TrackerGOTURN_create()
if tracker_type == "CSRT":
    tracker = cv2.TrackerCSRT_create()

# Define an initial bounding box
bbox = (0, 200, 330, 815)
# Uncomment the line below to select a different bounding box
ok,frame = capture.read()
if not ok:
    print
    'Cannot read video file'
    sys.exit()

#bbox = cv2.selectROI(frame, False)
#print(bbox)
#########################################################

while True:
    if iteration != 1:
        ret, frame = capture.read()

    if frame is None:
        break

    if iteration==1:
       first_frame = frame

    ######tracker
    # Initialize tracker with first frame and bounding box
    ok = tracker.init(frame, bbox)
    # Update tracker
    ok, bbox = tracker.update(frame)

    # Calculate Frames per second (FPS)
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);

    # Draw bounding box
    #if ok:
        # Tracking success
        #p1 = (int(bbox[0]), int(bbox[1]))
        #p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        #cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
    #else:
        # Tracking failure
        # cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    # Display tracker type on frame
    cv2.putText(frame, tracker_type + " Tracker", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);

    # Display FPS on frame
    cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);

    ##############
    #frame = cv2.medianBlur(frame, 5)
    #frame = cv2.bilateralFilter(frame, 9, 75, 75)
    #cv.imshow('real Frame 1', frame)
    #frame = 255 - cv2.absdiff(frame,background)
    #cv.imshow('real Frame 2', frame)
    #kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    #frame = cv.filter2D(frame, -1, kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    frame = cv2.dilate(frame, kernel, iterations=2)  # wider

    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #frame[:, 1, :] = 1.5*frame[:, 1, :]
    #frame[:, :, 1] = 1.5 * frame[:, :, 1]
    #frame[1, :, :] = 1.5 * frame[1, :, :]
    #frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    #frame = cv.equalizeHist(frame)
    #cv.imshow('equalizeHist', frame)

    # convert image from RGB to HSV
    #frame_hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    # Histogram equalisation on the V-channel
    #frame_hsv[:, :, 2] = cv2.equalizeHist(frame_hsv[:, :, 2])
    # convert image back from HSV to RGB
    #frame = cv2.cvtColor(frame_hsv, cv2.COLOR_HSV2RGB)



    ## [apply]
    #update the background model
    # learningRate = 1 the background model is completely reinitialized from the last frame.
    # learningRate = between 0 and 1 that indicates how fast the background model is learnt.
    # learningRate = -1 some automatically chosen learning rate
    #frame = frame[int(bbox[1]):int(bbox[1]+bbox[3]),int(bbox[0]):int(1.1*(bbox[0]+bbox[2])), : ]
    fgMask = backSub.apply(frame,learningRate = -1) # learningrate
    fgMask[:,0:int(bbox[0])] = 0
    fgMask[0:int(bbox[1]),:] = 0
    fgMask[:,int(bbox[0]+bbox[2]):] = 0
    fgMask[int(bbox[1]+bbox[3]):,:] = 0
    ## [apply]

    ## [display_frame_number]
    #get the frame number and write it on the current frame
    #cv.rectangle(frame, (10, 2), (100,20), (255,255,255), -1)
    #cv.putText(frame, str(capture.get(cv.CAP_PROP_POS_FRAMES)), (15, 15),
    #           cv.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
    ## [display_frame_number]


    #yonatan's add
    ###############################################################################################
    #if (iteration <= int(n_frames / 4)): # 0 to 1
    #    fgMask[:, int(1.8*1 * fgMask.shape[1]/4):int(4*fgMask.shape[1]/4)]=0

    #if (iteration <= int(2*n_frames / 4) and (iteration >= 1*n_frames/4)): #1 to 2
    #    fgMask[:, int(0 * fgMask.shape[1]/4):int(0.8*1*fgMask.shape[1]/4)]=0
    #    fgMask[:, int(1.4*2 * fgMask.shape[1]/4):int(4*fgMask.shape[1]/4)]=0

    #if (iteration <= 3*n_frames / 4) and (iteration >= 2*n_frames/4): #2 to 3
    #    fgMask[:, int(0 * fgMask.shape[1]/4):int(0.8*2*fgMask.shape[1]/4)]=0
    #    fgMask[:, int(1.4*3 * fgMask.shape[1]/4):int(4*fgMask.shape[1]/4)]=0

    #if (iteration <= 4*n_frames / 4) and (iteration >= 3*n_frames / 4): #3 to 4
    #    fgMask[:, int(0 * fgMask.shape[1]):int(0.8*3* fgMask.shape[1] / 4)]=0
    ##############################################################################################

    #kernel = np.ones((5,5),np.uint8)
    #kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))
    fgMask[fgMask < 254] = 0
    #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    #fgMask = cv2.erode(fgMask, kernel, iterations=1)  # thiner

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2))
    fgMask = cv2.erode(fgMask, kernel, iterations=4)  # thiner
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2 ))
    fgMask = cv2.dilate(fgMask, kernel, iterations=2)  # wider

    #fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_OPEN, kernel)
    #fgMask[fgMask < 150] = 0

    #fgMask = cv2.bilateralFilter(fgMask, 9, 250, 250)
    #fgMask = cv2.dilate(fgMask, kernel, iterations=4)  # wider

    #fgMask = cv2.medianBlur(fgMask, 21)
    fgMask = fgMask/255
    #frame = fgMask * frame #for grayscale images
    frame[:, :, 0] = fgMask * frame[:, :, 0]
    frame[:, :, 1] = fgMask * frame[:, :, 1]
    frame[:, :, 2] = fgMask * frame[:, :, 2]
    #hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    #hsv[:,:,1] = hsv[:,:,1]*fgMask
    #frame = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    #print(frame.shape)  (1080,1920,3)

    ## [show]
    #show the current frame and the fg masks
    cv.imshow('Frame', frame)
    cv.imshow('FG Mask', fgMask)
    iteration = iteration + 1
    ## [show]
    out.write(frame)
    fgMask = np.uint8(255*fgMask)
    fgMask = cv2.cvtColor(fgMask, cv2.COLOR_GRAY2RGB)
    out_bin.write(fgMask)
    print('processing frame number',iteration)


    keyboard = cv.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break

# Release video
capture.release()
out.release()
out_bin.release()