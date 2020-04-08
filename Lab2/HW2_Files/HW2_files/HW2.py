import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.io as sio

# FILL IN YOUR ID
# ID1 = 123456789
# ID2 = 987654321


#####################################PART 1: Lucas-Kanade Optical Flow################################################

# Load images I1,I2
IMG = sio.loadmat('HW2_PART1_IMAGES.mat')
I1 = IMG['I1']
I2 = IMG['I2']

# Choose parameters
WindowSize = None # Add your value here!
MaxIter = None # Add your value here!
NumLevels = None # Add your value here!

# Compute optical flow using LK algorithm
(u,v) = LucasKanadeOpticalFlow(I1,I2,WindowSize,MaxIter,NumLevels)

# Warp I2
I2_warp = WarpImage(I2,u,v)

# The RMS should decrease as the warped image (I2_warp) should be more similar to I1
print('RMS of original frames: '+ str(np.sum(np.sum(np.abs((I1-I2)**2)))))
print('RMS of processed frames: ' + str(np.sum(np.sum(np.abs((I1-I2_warp)**2)))))


# Plot I1,I2,I2_warp




###########################################3PART 2: Video Stabilization################################################

# Choose parameters
WindowSize = None # Add your value here!
MaxIter= None # Add your value here!
NumLevels = None # Add your value here!

#Load video file
InputVidName='input.avi'

# Stabilize video - save the stabilized video inside the function
StabilizedVid = LucasKanadeVideoStabilization(InputVid,WindowSize,MaxIter,NumLevels)
