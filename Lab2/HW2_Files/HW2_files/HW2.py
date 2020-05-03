import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.io as sio
from ex2_functions import *

# FILL IN YOUR ID
# ID1 = 204356315
# ID2 = 200846038

#####################################PART 2: Lucas-Kanade Optical Flow################################################

# Load images I1,I2
IMG = sio.loadmat('HW2_PART1_IMAGES.mat')
I1 = IMG['I1']
I2 = IMG['I2']

# Choose parameters
WindowSize = 5
MaxIter = 6
NumLevels = 4

# Compute optical flow using LK algorithm
(u,v) = LucasKanadeOpticalFlow(I1,I2,WindowSize,MaxIter,NumLevels)

# Warp I2
I2_warp = WarpImage(I2,u,v)

# The RMS should decrease as the warped image (I2_warp) should be more similar to I1
print('RMS of original frames: '+ str(np.sum(np.sum(np.abs((I1-I2)**2)))))
print('RMS of processed frames: ' + str(np.sum(np.sum(np.abs((I1-I2_warp)**2)))))

#for debug
#plt.imshow(I1)
#plt.imshow(I2)
#plt.imshow(I2_warp)

###########################################PART 3: Video Stabilization################################################

# Choose parameters
WindowSize = 5
MaxIter= 6
NumLevels = 4

#Load video file
InputVidName='input.avi'

# Stabilize video - save the stabilized video inside the function
StabilizedVid = LucasKanadeVideoStabilization(InputVidName,WindowSize,MaxIter,NumLevels)