import cv2 as cv
# from scipy import stats
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
from Code.compute_distance_map import *


###############################################################################



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
    seeds = plt.ginput(10)
    seeds = [(int(i[1]),int(i[0])) for i in seeds]

    #seeds= []
    #for i in seeds_input:
    #    if seeds_input[int(i[1]),int(i[0])] == 100:
    #        seeds.append((int(i[1]),int(i[0])))

    print("clicked", seeds)
    plt.show()


    # scaling the input image influences the weight between
    # euclidean and intensity distance in the final distance map
    im = normalize_im(im) * 255.

    # compute distance map
    dm = compute_dm_rasterscan(im, seeds, its=iterations, dist_type=dist_type)

    #plot image
    plt.imshow(dm,cmap='gray')
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
background = cv.imread('background.jpg')
background_hsv = cv.cvtColor(background, cv.COLOR_RGB2HSV)
background_v = background_hsv[:,:,2]

#read background scribbles
background_scrib = cv.imread('Background_full_scrib.jpg')
gray_back_scrib = cv.cvtColor(background_scrib, cv.COLOR_BGR2GRAY)#29=scrib, 255=garbage
gray_back_scrib[gray_back_scrib<=170]=100
gray_back_scrib[gray_back_scrib>=171]=0

#read foreground scribbles
foreground_scrib = cv.imread('Foreground_full_scrib.jpg')
gray_fore_scrib = cv.cvtColor(foreground_scrib, cv.COLOR_BGR2GRAY)#150=scrib, 255=garbage
gray_fore_scrib[gray_fore_scrib<=170]=100#!=255
gray_fore_scrib[gray_fore_scrib>=171]=0#==255

#plot background and foreground scribbels
cv.imshow("test",background_scrib)
cv.imshow("test",foreground_scrib)

#########################################################
# Start timer
timer = cv.getTickCount()

    # Set up tracker.
    # Instead of MIL, you can also use

tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'CSRT']
tracker_type = tracker_types[2]

if tracker_type == 'BOOSTING':
    tracker = cv.TrackerBoosting_create()
if tracker_type == 'MIL':
    tracker = cv.TrackerMIL_create()
if tracker_type == 'KCF':
    tracker = cv.TrackerKCF_create()
if tracker_type == 'TLD':
    tracker = cv.TrackerTLD_create()
if tracker_type == 'MEDIANFLOW':
    tracker = cv.TrackerMedianFlow_create()
if tracker_type == 'GOTURN':
    tracker = cv.TrackerGOTURN_create()
if tracker_type == "CSRT":
    tracker = cv.TrackerCSRT_create()

# Define an initial bounding box
bbox = (0, 200, 330, 815)


iteration = 0
while True:
    iteration = iteration + 1
    ret, frame_stab = capture_stab.read()
    ret, frame_bin = capture_bin.read()
    frame_stab_final = cv.cvtColor(frame_stab, cv.COLOR_RGB2HSV)  # 29=scrib, 255=garbage
    frame_bin = cv.cvtColor(frame_bin, cv.COLOR_RGB2GRAY)  # 29=scrib, 255=garbage

    if frame_stab is None:
        break
    if frame_bin is None:
        break

    ######tracker
    # Initialize tracker with first frame and bounding box
    ok = tracker.init(frame_stab, bbox)
    # Update tracker
    ok, bbox = tracker.update(frame_stab)

    # Calculate Frames per second (FPS)
    fps = cv.getTickFrequency() / (cv.getTickCount() - timer);

    # Display tracker type on frame
    cv.putText(frame_stab, tracker_type + " Tracker", (100, 20), cv.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);

    # Display FPS on frame
    cv.putText(frame_stab, "FPS : " + str(int(fps)), (100, 50), cv.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);


    frame_stab_tracked = frame_stab[int(bbox[1]):int((bbox[1] + bbox[3])),int(bbox[0]):int(1.3*(bbox[0] + bbox[2])),:]
    frame_bin_tracked = frame_bin[int(bbox[1]):int((bbox[1] + bbox[3])),int(bbox[0]):int(1.3*(bbox[0] + bbox[2]))]
    gray_back_scrib = gray_back_scrib[int(bbox[1]):int((bbox[1] + bbox[3])),int(bbox[0]):int(1.3*(bbox[0] + bbox[2]))]
    gray_fore_scrib = gray_fore_scrib[int(bbox[1]):int((bbox[1] + bbox[3])),int(bbox[0]):int(1.3*(bbox[0] + bbox[2]))]

    #convert frame to hsv
    #frame_stab = cv.bilateralFilter(frame_stab, 18, 75, 75)

    frame_stab_hsv_tracked = cv.cvtColor(frame_stab_tracked, cv.COLOR_RGB2HSV)
    frame_stab_tracked = frame_stab_hsv_tracked[:,:,2]

    #frame_stab = cv.equalizeHist(frame_stab)
    #kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    #frame_stab = cv.filter2D(frame_stab, -1, kernel)
    #frame_stab = cv.medianBlur(frame_stab, 5)

    #takes only points from scribble
    if iteration==1:
        print(type(frame_stab))
        background_array = frame_stab_tracked[gray_back_scrib == 100]
        foreground_array = frame_stab_tracked[gray_fore_scrib == 100]

    #binary maske normalized to [0,1]
    frame_bin = frame_bin/255

    #calculate KDE for background and foreground
    x_grid = np.linspace(0, 255, 256)
    kde_back = gaussian_kde(background_array.ravel())#, bw_method='silverman')
    kde_back_pdf = kde_back.evaluate(x_grid)
    kde_fore = gaussian_kde(foreground_array.ravel())#, bw_method='silverman')
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
    Probability_map_fore = P_F[frame_stab_tracked]
    Probability_map_back = P_B[frame_stab_tracked]


    #plot probabilty maps
    #cv.imshow("Probability map foreground", Probability_map_fore);
    #cv.imshow("Probability map background", Probability_map_back);

    #cv.namedWindow("Probability map foreground", cv.WND_PROP_FULLSCREEN)
    #cv.setWindowProperty("Probability map foreground", cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)


    #resize image
    #Probability_map_fore = cv.resize(Probability_map_fore,(int(Probability_map_fore.shape[1]/2),int(Probability_map_fore.shape[0]/2)), interpolation=cv.INTER_AREA)
    #Probability_map_back = cv.resize(Probability_map_back,(int(Probability_map_back.shape[1]/2),int(Probability_map_back.shape[0]/2)), interpolation=cv.INTER_AREA)

    #compute gaodesic distance
    print(Probability_map_fore.shape)
    Seeds_foreground = [(59, 187), (221, 149), (261, 202), (368, 173), (432, 238), (492, 167), (730, 89), (741, 327), (359, 89), (283, 285)]
    Seeds_background = [(52, 60), (330, 36), (568, 62), (697, 156), (659, 356), (468, 336), (154, 360), (30, 316), (793, 316), (777, 38)]
    #gaodesic_fore = gaodesic_ditance(Probability_map_fore,Seeds_foreground);
    #gaodesic_back = gaodesic_ditance(Probability_map_back,Seeds_background);
    gaodesic_fore = gaodesic_ditance(Probability_map_fore,gray_fore_scrib);
    gaodesic_back = gaodesic_ditance(Probability_map_back,gray_back_scrib);
    print('done')

    foreground_tracked = np.zeros((frame_stab_tracked.shape))
    foreground_tracked[gaodesic_fore-gaodesic_back<=0] = frame_stab_tracked[gaodesic_fore-gaodesic_back<=0]


    Vforeground = frame_stab_tracked
    Vforeground = (gaodesic_fore - gaodesic_back >= 0) * np.zeros(Vforeground.shape)
    Vforeground = (gaodesic_fore - gaodesic_back < 0) * np.ones(Vforeground.shape)
    #Vforeground is now binary map of the foreground
    #cv.imshow('Vforground',Vforeground)

    #check where is the outline of the foreground
    delta_edges = bwperim(Vforeground, n=4) #binary image
    #cv.imshow('delta_edges',delta_edges)


    #do dialation with radius roo around the foreground edges
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2, 2))
    B_ro = cv.dilate(delta_edges, kernel, iterations=2)  # wider binary image
    B_ro = (B_ro==1) * 0.5
    #cv.imshow('B_ro',B_ro)


    trimap = np.zeros((Vforeground.shape))
    trimap = Vforeground#foreground area filled with '1'
    trimap[B_ro == 0.5] =  0.5# undecided area
    #all background area is filled with zeros

    cv.imshow('trimap',trimap)

    #alpha mating
    r = 1 # 0<r<2   -can change to desired value
    r = r * np.ones((frame_stab_tracked.shape))
    epsilon = 0.00001
    gaodesic_fore[gaodesic_fore==0] = epsilon # dont divide by zero
    gaodesic_back[gaodesic_back == 0] = epsilon
    w_fore = cv.multiply(np.power(gaodesic_fore,-r),Probability_map_fore)
    w_back = cv.multiply(np.power(gaodesic_back,-r),Probability_map_back)

    ####debug####
    #print(frame_stab.shape)
    #print(Probability_map_fore.shape)
    #print(w_fore.shape)
    #print(w_back.shape)
    #print(frame_stab.shape)
    #print(gaodesic_fore.shape)
    #print(int(bbox[1]))
    #print(int(bbox[1] + bbox[3]))
    #print(int(bbox[0]))
    #print(int(1.3 * (bbox[0] + bbox[2])))
    #test = w_fore / (w_fore+w_back)
    #print((test).shape)
    ####

    #crates alpha map
    alpha = np.zeros((frame_stab_tracked.shape))
    alpha[trimap == 0.5] = w_fore[trimap == 0.5] / (w_fore[trimap == 0.5] + w_back[trimap == 0.5]) #blending
    alpha[trimap == 0  ] = 0 #background
    alpha[trimap == 1  ] = 1 #foreground


    # Perform alpha blending
    foreground_mask_tracked = np.zeros(foreground_tracked.shape)
    mask = foreground_tracked>0
    foreground_mask_tracked= mask * np.ones(mask.shape)
    foreground = np.zeros(frame_stab_final.shape)
    foreground[int(bbox[1]): int(bbox[1] + bbox[3]), int(bbox[0]): int(1.3 * (bbox[0] + bbox[2])),0] =  foreground_mask_tracked * frame_stab_final[int(bbox[1]): int(bbox[1] + bbox[3]), int(bbox[0]): int(1.3 * (bbox[0] + bbox[2])),0]
    foreground[int(bbox[1]): int(bbox[1] + bbox[3]), int(bbox[0]): int(1.3 * (bbox[0] + bbox[2])),1] =  foreground_mask_tracked * frame_stab_final[int(bbox[1]): int(bbox[1] + bbox[3]), int(bbox[0]): int(1.3 * (bbox[0] + bbox[2])),1]
    foreground[int(bbox[1]): int(bbox[1] + bbox[3]), int(bbox[0]): int(1.3 * (bbox[0] + bbox[2])),2] =  foreground_tracked
    background = background_hsv

    alpha_full = np.zeros((background_hsv.shape))
    alpha_full[int(bbox[1]): int(bbox[1] + bbox[3]), int(bbox[0]): int(1.3 * (bbox[0] + bbox[2])),0] =  alpha
    alpha_full[int(bbox[1]): int(bbox[1] + bbox[3]), int(bbox[0]): int(1.3 * (bbox[0] + bbox[2])),1] =  alpha
    alpha_full[int(bbox[1]): int(bbox[1] + bbox[3]), int(bbox[0]): int(1.3 * (bbox[0] + bbox[2])),2] =  alpha

    alpha_full=alpha_full.astype(float)
    foreground_mul_alpha = cv.multiply(alpha_full, foreground)

    x=np.ones((alpha_full.shape)) - alpha_full
    y=background
    background_mul= cv.multiply(x.astype(float) , y.astype(float))

    background_mul = background_mul.astype(int)
    foreground_mul_alpha = foreground_mul_alpha.astype(int)
    foreground = foreground.astype(float)

    outImage = cv.add(background_mul, foreground_mul_alpha)
    outImage = np.uint8(outImage)
    outImage = cv.cvtColor(outImage, cv.COLOR_HSV2RGB)

    # write output matted video
    out.write(outImage)
    cv.imshow('Matted', outImage)

# Release video
capture_stab.release()
capture_bin.release()
out.release()
