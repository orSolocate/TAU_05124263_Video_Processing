import numpy as np
import matplotlib.pyplot as plt
from unused.compute_distance_map import *


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
