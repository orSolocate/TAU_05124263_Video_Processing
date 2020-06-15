import cv2 as cv
import numpy as np
# from scipy import stats
import matplotlib.pyplot as plt
from scipy.stats.distributions import norm
from scipy.stats import gaussian_kde
from scipy import stats
import matplotlib.pyplot as plt
from compute_distance_map import *

###############################################################################
def normalize_im(I):
    normed_im = (1. * I - np.amin(I))/(np.amax(I)-np.amin(I))
    return normed_im


def gaodesic_ditance(I):
    dist_type = 'geodesic'
    iterations = 2

    # load intensity image
    im = I  #########################################################

    # define foreground
    #seeds = np.transpose(np.where(dotim > 0.5))
    plt.imshow(im)
    print("Please click")
    seeds = plt.ginput(10)
    seeds = [(int(i[1]),int(i[0])) for i in seeds]#risky
    print("clicked", seeds)
    plt.show()

    # scaling the input image influences the weight between
    # euclidean and intensity distance in the final distance map
    im = normalize_im(im) * 255.

    # compute distance map
    dm = compute_dm_rasterscan(im, seeds, its=iterations, dist_type=dist_type)

    #plot image
    plt.imshow(dm)
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.show()
    return dm

###############################################################################

## [capture]
capture_stab = cv.VideoCapture('stabilize.avi')
capture_bin = cv.VideoCapture('binary.avi')

if not capture_stab.isOpened:
    print('Unable to open video')
    exit(0)

if not capture_bin.isOpened:
    print('Unable to open video')
    exit(0)

# Get width and height of video stream
w = int(capture_stab.get(cv.CAP_PROP_FRAME_WIDTH))
h = int(capture_stab.get(cv.CAP_PROP_FRAME_HEIGHT))

# Get frames per second (fps)
fps = capture_stab.get(cv.CAP_PROP_FPS)

# Define the codec for output video
fourcc = cv.VideoWriter_fourcc(*'MJPG')

# Set up output video
out = cv.VideoWriter('matted.avi', fourcc, fps, (w, h))

# read background image
background = cv.imread('background.jpeg')

#read background scribbles
background_scrib = cv.imread('Background_scrib.jpg')
gray_back_scrib = cv.cvtColor(background_scrib, cv.COLOR_BGR2GRAY)#29=scrib, 255=garbage
gray_back_scrib[gray_back_scrib<=170]=100
gray_back_scrib[gray_back_scrib>=171]=0

#read foreground scribbles
foreground_scrib = cv.imread('Foreground_scrib.jpg')
gray_fore_scrib = cv.cvtColor(foreground_scrib, cv.COLOR_BGR2GRAY)#150=scrib, 255=garbage
gray_fore_scrib[gray_fore_scrib<=170]=100#!=255
gray_fore_scrib[gray_fore_scrib>=171]=0#==255

#plot background and foreground scribbels
#cv.imshow("test",background_scrib)
#cv.imshow("test",foreground_scrib)

iteration = 0
while True:
    iteration = iteration + 1
    ret, frame_stab = capture_stab.read()
    ret, frame_bin = capture_bin.read()

    if frame_stab is None:
        break
    if frame_bin is None:
        break

    #convert frame to hsv
    frame_stab_hsv = cv.cvtColor(frame_stab, cv.COLOR_RGB2HSV)
    frame_stab = frame_stab_hsv[:,:,2]

    #takes only points from scribble
    if iteration==1:
        background_array = frame_stab[gray_back_scrib == 100]
        foreground_array = frame_stab[gray_fore_scrib == 100]

    #binary maske normalized to [0,1]
    frame_bin = frame_bin/255

    #calculate KDE for background and foreground
    x_grid = np.linspace(0, 255, 256)
    kde_back = gaussian_kde(background_array.ravel(), bw_method='silverman')
    kde_back_pdf = kde_back.evaluate(x_grid)
    kde_fore = gaussian_kde(foreground_array.ravel(), bw_method='silverman')
    kde_fore_pdf = kde_fore.evaluate(x_grid)

    #plot KDE  from all picture
    plt.figure()
    plt.plot(x_grid, kde_back_pdf, label='kde background', color="g")
    plt.title('Kernel Density Estimation - Background')
    plt.legend()
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.show()

    plt.figure()
    plt.plot(x_grid, kde_fore_pdf, label='kde foreground', color="g")
    plt.title('Kernel Density Estimation - Foreground')
    plt.legend()
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.show()

    #probabilties of background and foreground
    P_F = kde_fore_pdf / (kde_fore_pdf + kde_back_pdf)
    P_B = kde_back_pdf / (kde_fore_pdf + kde_back_pdf)

    #probabilies map of background and foreground
    Probability_map_fore = P_F[frame_stab]
    Probability_map_back = P_B[frame_stab]

    #plot probabilty maps
    cv.imshow("Probability map foreground", Probability_map_fore);
    #cv.namedWindow("Probability map foreground", cv.WND_PROP_FULLSCREEN)
    #cv.setWindowProperty("Probability map foreground", cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
    cv.imshow("Probability map background", Probability_map_back);

    #resize image
    Probability_map_fore = cv.resize(Probability_map_fore,(int(Probability_map_fore.shape[1]/2),int(Probability_map_fore.shape[0]/2)), interpolation=cv.INTER_AREA)
    Probability_map_back = cv.resize(Probability_map_back,(int(Probability_map_back.shape[1]/2),int(Probability_map_back.shape[0]/2)), interpolation=cv.INTER_AREA)

    #compute gaodesic distance
    gaodesic_fore = gaodesic_ditance(Probability_map_fore);
    gaodesic_back = (Probability_map_back);
    print('done')

    # write output matted video
    # out.write(frame_matted)

# Release video
capture_stab.release()
capture_bin.release()
out.release()
