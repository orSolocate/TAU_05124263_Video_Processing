import numpy as np
import GeodisTK
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import cv2
from unused.compute_distance_map import *


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
        cv2.imshow('Fast Marching Geodesic distance',D1)
    return D1

def fixBorder_inverse(frame):
    s = frame.shape
    T = cv2.getRotationMatrix2D((s[1] / 2, s[0] / 2), 0, 1.08)
    T = cv2.invertAffineTransform(T)
    frame = cv2.warpAffine(frame, T, (s[1], s[0]))
    return frame
