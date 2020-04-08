#import os
#from ex1_part2_functions import *
from ex1_Q3_functions import *
#import matplotlib.pyplot as plt #used just to save the image...

# FILL IN YOUR ID
# ID1 = 200846038
# ID2 = 204356315

###############################################Part_3#######################################################

# LOAD CHECKERBOARD IMAGE
I1 = cv2.imread('I1.jpg')  # Read image as grayscale
# LOAD ATTACHED IMAGE
I2 = cv2.imread('I2.jpg')

# Harris Corner Detector Parameters, you may change them
K = 0.06
FIRST_THRESHOLD = 20000000
SECOND_THRESHOLD =41000000
use_grid = True

# CALL YOUR FUNCTION TO FIND THE CORNER PIXELS
I1_CORNERS = myHarrisCornerDetector(I1, K, FIRST_THRESHOLD, use_grid)
I2_CORNERS = myHarrisCornerDetector(I2, K, SECOND_THRESHOLD, use_grid)

# CALL YOUR FUNCTION TO CREATE THE PLOT
createCornerPlots(I1, I1_CORNERS, I2, I2_CORNERS)

# SAVE FIGURE
#plt.savefig(os.getcwd()+'\\ex1_200846038_204356315.jpg')