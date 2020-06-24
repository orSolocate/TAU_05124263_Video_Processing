import cv2 as cv
#import numpy as np
# from scipy import stats
#import matplotlib.pyplot as plt
#from scipy.stats.distributions import norm
from scipy.stats import gaussian_kde
#from scipy import stats
import matplotlib.pyplot as plt
#import numpy as np
from compute_distance_map import *
#import sys
import config
from tqdm import tqdm
import logging
import GeodisTK
import video_handling
##############################################################################

def bwperim(bw, n=4):
    """
    perim = bwperim(bw, n=4)
    Find the perimeter of objects in binary images.
    A pixel is part of an object perimeter if its value is one and there
    is at least one zero-valued pixel in its neighborhood.
    By default the neighborhood of a pixel is 4 nearest pixels, but
    if `n` is set to 8 the 8 nearest pixels will be considered.
    Parameters
    ----------
      bw : A black-and-white image
      n : Connectivity. Must be 4 or 8 (default: 8)
    Returns
    -------
      perim : A boolean image
    """

    if n not in (4,8):
        raise ValueError('mahotas.bwperim: n must be 4 or 8')
    rows,cols = bw.shape

    # Translate image by one pixel in all directions
    north = np.zeros((rows,cols))
    south = np.zeros((rows,cols))
    west = np.zeros((rows,cols))
    east = np.zeros((rows,cols))

    north[:-1,:] = bw[1:,:]
    south[1:,:]  = bw[:-1,:]
    west[:,:-1]  = bw[:,1:]
    east[:,1:]   = bw[:,:-1]
    idx = (north == bw) & \
          (south == bw) & \
          (west  == bw) & \
          (east  == bw)
    if n == 8:
        north_east = np.zeros((rows, cols))
        north_west = np.zeros((rows, cols))
        south_east = np.zeros((rows, cols))
        south_west = np.zeros((rows, cols))
        north_east[:-1, 1:]   = bw[1:, :-1]
        north_west[:-1, :-1] = bw[1:, 1:]
        south_east[1:, 1:]     = bw[:-1, :-1]
        south_west[1:, :-1]   = bw[:-1, 1:]
        idx &= (north_east == bw) & \
               (south_east == bw) & \
               (south_west == bw) & \
               (north_west == bw)
    return ~idx * bw
###############################################################################
def normalize_im(I):
    normed_im = (1. * I - np.amin(I))/(np.amax(I)-np.amin(I))
    return normed_im


def gaodesic_ditance(I,seeds_input):
    dist_type = 'geodesic'
    iterations = 1

    # load intensity image
    im = I  #########################################################

    # define foreground
    plt.imshow(im)
    print("Please click")
    #seeds = plt.ginput(10)
    #seeds = [(int(i[1]),int(i[0])) for i in seeds]

    seeds= []
    for i in seeds_input:
        if seeds_input[int(i[1]),int(i[0])] == 100:
            seeds.append((int(i[1]),int(i[0])))

    print("clicked", seeds)
    plt.figure(3)
    plt.show(block=False)


    # scaling the input image influences the weight between
    # euclidean and intensity distance in the final distance map
    im = normalize_im(im) * 255.

    # compute distance map
    dm = compute_dm_rasterscan(im, seeds, its=iterations, dist_type=dist_type)

    #plot image
    plt.imshow(dm,cmap='gray')
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.figure(4)
    plt.show(block=False)
    return dm

###############################################################################

def kde_evaluate(array,isPlotted,title=''):
    x_grid = np.linspace(0, 255, 256)
    if (array.size==0):
        kde_pdf=np.zeros(256,dtype='float64')
    else:
        kde = gaussian_kde(array.ravel())  # , bw_method='silverman')
        kde_pdf = kde.evaluate(x_grid)
    if (isPlotted):
        plt.figure()
        plt.plot(x_grid, kde_pdf, color="g")
        plt.title(title)
        plt.legend()
        plt.show(block=False)
    return kde_pdf


def geo_distance(frame, scrible_pos, isPlotted):
    I = np.asanyarray(frame, np.float32)
    S = np.zeros(I.shape, np.uint8)
    S[scrible_pos == 100] = 1
    D1 = GeodisTK.geodesic2d_fast_marching(I, S)
    if (isPlotted):
        cv.imshow('Fast Marching Geodesic distance',D1)
    return D1


def Matting():
    print("\nMatting:")
    if not (config.unstable):
        if (config.DEMO):
            capture_stab = cv.VideoCapture(config.demo_stabilized_vid_file)
        else:
            capture_stab = cv.VideoCapture(config.stabilized_vid_file)
        n_frames_stab,fourcc, fps, out_size=video_handling.extract_video_params(capture_stab)
        alpha_out = cv.VideoWriter(config.alpha_vid_file, fourcc, fps, out_size)
    else:
        capture_stab = cv.VideoCapture(config.in_vid_file)
        n_frames_stab,fourcc, fps, out_size=video_handling.extract_video_params(capture_stab)
        alpha_out = cv.VideoWriter(config.un_alpha_vid_file, fourcc, fps, out_size)

    capture_bin = cv.VideoCapture(config.binary_vid_file)
    n_frames_bin, fourcc_bin, fps_bin, out_size_bin = video_handling.extract_video_params(capture_bin)

    out = cv.VideoWriter(config.matted_vid_file, fourcc, fps,out_size)

    # read background image
    background = cv.imread(config.in_background_file)
    background_hsv = cv.cvtColor(background, cv.COLOR_BGR2HSV)

    print("\nprocess frames..")
    forePlotted = config.forePlotted
    backPlotted = config.backPlotted
    for iteration in tqdm(range(1,min(n_frames_stab,n_frames_bin)-config.MAT_frame_reduction_DEBUG)): #why do you start from 1??
        ret, frame_stab = capture_stab.read()
        ret, frame_bin = capture_bin.read()
        frame_stab_hsv = cv.cvtColor(frame_stab, cv.COLOR_BGR2HSV)  # 29=scrib, 255=garbage



        gray_fore_scrib = cv.cvtColor(frame_bin, cv.COLOR_BGR2GRAY)  # 29=scrib, 255=garbage
        gray_fore_scrib[gray_fore_scrib>=200]=100#!=255
        gray_fore_scrib[gray_fore_scrib<=99]=0#==255
        gray_back_scrib = cv.bitwise_not(cv.cvtColor(frame_bin, cv.COLOR_BGR2GRAY))
        gray_back_scrib[gray_back_scrib<=99]=0
        gray_back_scrib[gray_back_scrib>=200]=100

        # Display FPS on frame
        cv.putText(frame_stab, "FPS : " + str(int(fps)), (100, 50), cv.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

        frame_stab_hsv = cv.cvtColor(frame_stab, cv.COLOR_BGR2HSV)
        frame_stab = frame_stab_hsv[:,:,2]

        #takes only points from scribble
        background_array = frame_stab[gray_back_scrib == 100]
        foreground_array = frame_stab[gray_fore_scrib == 100]

        #binary maske normalized to [0,1]
        frame_bin = frame_bin/255

        #calculate KDE for background and foreground
        kde_fore_pdf=kde_evaluate(foreground_array,forePlotted,title='Kernel Density Estimation - Background')
        kde_back_pdf=kde_evaluate(background_array,backPlotted,title='Kernel Density Estimation - Foreground')

        #probabilties of background and foreground
        P_F = kde_fore_pdf / (kde_fore_pdf + kde_back_pdf)
        P_B = kde_back_pdf / (kde_fore_pdf + kde_back_pdf)

        #probabilies map of background and foreground
        Probability_map_fore = P_F[frame_stab]
        Probability_map_back = P_B[frame_stab]


        #New Geodesic!!!#
        gaodesic_fore=geo_distance(frame_stab, gray_fore_scrib, forePlotted)
        gaodesic_back=geo_distance(frame_stab, gray_back_scrib, backPlotted)

        if (config.plot_from_here<=iteration<=config.plot_until_here):
            forePlotted=True
            backPlotted=True
        else:
            forePlotted = config.forePlotted
            backPlotted = config.backPlotted

        foreground = np.zeros((frame_stab.shape))
        foreground[gaodesic_fore-gaodesic_back<=0] = frame_stab[gaodesic_fore-gaodesic_back<=0]

        Vforeground = frame_stab
        Vforeground = (gaodesic_fore - gaodesic_back >= 0) * np.zeros(Vforeground.shape)
        Vforeground = (gaodesic_fore - gaodesic_back < 0) * np.ones(Vforeground.shape)
        #Vforeground is now binary map of the foreground
        #cv.imshow('Vforground',Vforeground)

        #check where is the outline of the foreground
        delta_edges = bwperim(Vforeground, n=config.bwperim['n']) #binary image
        #cv.imshow('delta_edges',delta_edges)


        #do dialation with radius roo around the foreground edges
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, config.MAT_dilate['struct_size'])
        B_ro = cv.dilate(delta_edges, kernel, iterations=config.MAT_dilate['iterations'])  # wider binary image
        B_ro = (B_ro==1) * 0.5
        #cv.imshow('B_ro',B_ro)


        trimap = np.zeros((Vforeground.shape))
        trimap = Vforeground#foreground area filled with '1'
        trimap[B_ro == 0.5] =  0.5# undecided area
        #all background area is filled with zeros

        #cv.imshow('trimap',trimap)

        #alpha mating
        r = config.r * np.ones((frame_stab.shape))
        gaodesic_fore[gaodesic_fore==0] = config.epsilon # dont divide by zero
        gaodesic_back[gaodesic_back == 0] = config.epsilon
        w_fore = cv.multiply(np.power(gaodesic_fore,-r),Probability_map_fore)
        w_back = cv.multiply(np.power(gaodesic_back,-r),Probability_map_back)

        #crates alpha map
        alpha = np.zeros((frame_stab.shape))
        alpha[trimap == 0.5] = w_fore[trimap == 0.5] / (w_fore[trimap == 0.5] + w_back[trimap == 0.5]) #blending
        alpha[trimap == 0  ] = 0 #background
        alpha[trimap == 1  ] = 1 #foreground


        # Perform alpha blending
        mask = foreground>0
        foreground = np.zeros(frame_stab_hsv.shape)
        foreground_mask= mask * np.ones(mask.shape)
        print(foreground.shape,frame_stab_hsv.shape)
        foreground[:, :, 0] = foreground_mask * frame_stab_hsv[:, :, 0]
        foreground[:, :, 1] = foreground_mask * frame_stab_hsv[:, :, 1]
        foreground[:, :, 2] = foreground_mask * frame_stab_hsv[:, :, 2]
        background = background_hsv

        alpha_full = np.zeros((background_hsv.shape))
        alpha_full[:,:,0] =  alpha
        alpha_full[:,:,1] =  alpha
        alpha_full[:,:,2] =  alpha

        alpha_full_out = np.uint8(alpha_full * 255)
        alpha_out.write(alpha_full_out)

        alpha_full=alpha_full.astype(float)
        foreground_mul_alpha = cv.multiply(alpha_full, foreground)

        x=np.ones((alpha_full.shape)) - alpha_full
        y=background
        background_mul= cv.multiply(x.astype(float) , y.astype(float))

        background_mul = background_mul.astype(int)
        foreground_mul_alpha = foreground_mul_alpha.astype(int)

        outImage = cv.add(background_mul, foreground_mul_alpha)
        outImage = np.uint8(outImage)
        outImage = cv.cvtColor(outImage, cv.COLOR_HSV2BGR)

        # write output matted video
        out.write(outImage)
        #cv.imshow('Matted', outImage)

    # Release video
    cv.destroyAllWindows()
    capture_stab.release()
    capture_bin.release()
    out.release()
    alpha_out.release()
    return
