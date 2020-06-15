import numpy as np
#import imageio
#import pydicom
import cv2 as cv
import matplotlib.pyplot as plt
from compute_distance_map import *

def normalize_im(I):
    normed_im = (1. * I - np.amin(I))/(np.amax(I)-np.amin(I))
    return normed_im

example = 'pvs' # or tumor

filename = 'example/' + example + '_input' + '.png'
dist_type = 'geodesic'
iterations = 2
save_im = False

# load intensity image
im = cv.imread(filename)#########################################################

# define foreground
if example == 'pvs':
    print('Running example with pvs')
    im = im[:, :, 0] # rgba image to gray scale image
    #dotim = imageio.imread(filename.replace('input', 'seeds'))#####################################
    dotim = cv.imread('example/pvs_seeds.png')#####################################
    dotim = dotim[:, :, 0] # rgba image to gray scale image
    seeds = np.transpose(np.where(dotim>0.5))
else:
    print('Running example with tumor')
    seeds = np.array([[363,186]])

# scaling the input image influences the weight between
# euclidean and intensity distance in the final distance map
im = normalize_im(im) * 255.

# compute distance map
dm = compute_dm_rasterscan(im, seeds, its=iterations, dist_type=dist_type)

if save_im:
    save_filename = '_'.join(('example/' + example, dist_type, 'distance_map.png'))
    plt.imsave(save_filename, dm, cmap='viridis')
    print('Saved as ' + save_filename)
else:
    plt.imshow(dm)
    plt.show()
