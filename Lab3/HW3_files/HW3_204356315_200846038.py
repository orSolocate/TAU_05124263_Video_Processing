import cv2
import numpy as np
import numpy.matlib
from HW3_functions import *
import os

"""
 HW 3, COURSE 0512-4263, TAU 2020
 PARTICLE FILTER TRACKING
 THE PURPOSE OF THIS ASSIGNMENT IS TO IMPLEMENT A PARTICLE FILTER TRACKER
 IN ORDER TO TRACK A RUNNING PERSON IN A SERIES OF IMAGES.
 IN ORDER TO DO THIS YOU WILL WRITE THE FOLLOWING FUNCTIONS:
 IMPORTANT - YOU WILL USE compNormHist TO UPDATE THE INDIVIDUAL WEIGHTS
 OF EACH PARTICLE. AFTER YOU'RE DONE WITH THIS YOU WILL NEED TO COMPUTE
 THE 100 NORMALIZED WEIGHTS WHICH WILL RESIDE IN VECTOR W (1x100)
 AND THE CDF (CUMULATIVE DISTRIBUTION FUNCTION, C. SIZED 1x100)
 NORMALIZING 100 WEIGHTS MEANS THAT THE SUM OF 100 WEIGHTS = 1
"""


ID1 = "204356315"
ID2 = "200846038"

ID = "HW3_{0}_{1}".format(ID1, ID2)
IMAGE_DIR_PATH = "{0}\\Images".format(os.getcwd())


# SET NUMBER OF PARTICLES
N = 100

# Initial Settings
s_initial = [297,    # x center
             139,    # y center
              16,    #DEFAULT 16 half width
              43,    #DEFAULT 43 half height
               0,    # velocity x
               0]    # velocity y

# CREATE INITIAL PARTICLE MATRIX 'S' (SIZE 6xN)
s_intial_array=np.matlib.repmat(s_initial, N, 1).transpose()
S = predictParticles(s_intial_array)

# LOAD FIRST IMAGE
I = cv2.imread(IMAGE_DIR_PATH + os.sep + "001.png")

# COMPUTE NORMALIZED HISTOGRAM
q = compNormHist(I, s_initial)

# COMPUTE NORMALIZED WEIGHTS (W) AND PREDICTOR CDFS (C)
# YOU NEED TO FILL THIS PART WITH CODE:
W=np.zeros(N)
for i in range(0,N):
    p = compNormHist(I, S[:,i])
    W[i]=compBatDist(p, q)
W=np.true_divide(W,np.sum(W))

C=compute_CDF(W)

images_processed = 1

# MAIN TRACKING LOOP
image_name_list = os.listdir(IMAGE_DIR_PATH)
for image_name in image_name_list[1:]:
    print(image_name)
    S_prev = S

    # LOAD NEW IMAGE FRAME
    image_path = IMAGE_DIR_PATH + os.sep + image_name
    I = cv2.imread(image_path)

    # SAMPLE THE CURRENT PARTICLE FILTERS
    S_next_tag = sampleParticles(S_prev, C)

    # PREDICT THE NEXT PARTICLE FILTERS (YOU MAY ADD NOISE
    S = predictParticles(S_next_tag)

    # COMPUTE NORMALIZED WEIGHTS (W) AND PREDICTOR CDFS (C)
    # YOU NEED TO FILL THIS PART WITH CODE:
    for i in range(0, N):
        p = compNormHist(I, S[:, i])
        W[i] = compBatDist(p, q)
    W = np.true_divide(W, np.sum(W))
    C = compute_CDF(W)

    # CREATE DETECTOR PLOTS
    #showParticles(I, S, W, images_processed, ID)############delete me############
    images_processed += 1
    if 0 == images_processed%10:
        showParticles(I, S, W, images_processed, ID)
