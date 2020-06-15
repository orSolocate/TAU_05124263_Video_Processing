from __future__ import print_function
import cv2 as cv
import cv2
import numpy as np
from matplotlib import pyplot as plt
import argparse




## [capture]
capture = cv.VideoCapture('Stabilized_Example_INPUT.avi')
#capture = cv.VideoCapture('video_out.avi')
if not capture.isOpened:
    print('Unable to open: ' + args.input)
    exit(0)

## [capture]
iteration= 1

n_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
print(n_frames)

while iteration<=n_frames-1:
    print(iteration)
    ret, frame = capture.read()
    if frame is None:
        break

    if iteration==1:
        frame_history_b = np.zeros((frame.shape[0], frame.shape[1], n_frames))
        frame_history_g = np.zeros((frame.shape[0], frame.shape[1], n_frames))
        frame_history_r = np.zeros((frame.shape[0], frame.shape[1], n_frames))
        print(frame_history_b.shape)

    frame_history_b[:, :, iteration] = frame[:, :, 2]
    frame_history_g[:, :, iteration] = frame[:, :, 1]
    frame_history_r[:, :, iteration] = frame[:, :, 0]
    iteration = iteration + 1


output_picture_b = np.median(frame_history_b, axis=-1)
output_picture_g = np.median(frame_history_g, axis=-1)
output_picture_r = np.median(frame_history_r, axis=-1)

#unsharp mask
background = np.dstack((output_picture_r,output_picture_g,output_picture_b))
gaussian_3 = cv2.GaussianBlur(background, (9,9), 10.0)
background = cv2.addWeighted(background, 1.5, gaussian_3, -0.5, 0, background)
cv2.imwrite('background.jpeg', background)

# Release video
capture.release()
# Close windows
cv2.destroyAllWindows()