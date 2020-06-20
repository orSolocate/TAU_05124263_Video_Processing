import logging
import os.path as osp

# ~~~ DEMO ~~~ #
DEMO = True  # True for given stabilized
unstable = False

# ~~~ file paths ~~~ #
cur_path = osp.split(osp.dirname(osp.abspath(__file__)))[0]
in_vid_file = osp.join(cur_path, 'Input', 'INPUT.avi')
stabilized_vid_file = osp.join(cur_path, 'Outputs', 'stabilize.avi')
demo_stabilized_vid_file = osp.join(cur_path, 'Outputs', 'Stabilized_Example_INPUT.avi')
extracted_vid_file = osp.join(cur_path, 'Outputs', 'extracted.avi')
binary_vid_file = osp.join(cur_path, 'Outputs', 'binary.avi')
in_background_file = osp.join(cur_path, 'Input', 'background.jpg')
matted_vid_file = osp.join(cur_path, 'Outputs', 'matted.avi')
alpha_vid_file = osp.join(cur_path, 'Outputs', 'alpha.avi')
un_alpha_vid_file = osp.join(cur_path, 'Outputs', 'unstabilized_alpha.avi')
out_vid_file = osp.join(cur_path, 'Outputs', 'OUTPUT.avi')

# ~~~ logger ~~~ #
log_file = osp.join(cur_path, 'Outputs', 'RunTimeLog.txt')
logging.basicConfig(filename=log_file, filemode='w', level=logging.INFO)  # for debug: level=logging.DEBUG

# ~~~ Video_Stabilization ~~~ #
SMOOTHING_RADIUS = 150  # was 50 #mine 75  The larger the more stable the video, but less reactive to sudden panning

goodFeaturesToTrack = dict(maxCorners=1000,  # was 200 #mine 1000
                           qualityLevel=0.05,  # was 0.01
                           minDistance=10,  # was 30
                           blockSize=3)
calcOpticalFlowPyrLK = dict(winSize=(50, 50),
                            maxLevel=10)

# ~~~ Background_Substraction ~~~ #
BS_frame_reduction_DEBUG = 0
createBackground_Substraction = dict(history=170,
                                     dist2Threshold=500.0,
                                     detectShadows=True)
backSub_apply = dict(learningRate=-1)  # negative means automatic learningRate
BS_erode = dict(iterations=2, struct_size=(15, 10))  # thiner 1, (20,10)
BS_dilate = dict(iterations=5, struct_size=(4, 4))  # wider 5,  (5,5)
blob_detect=dict(minThreshold=0, # 10
                 maxThreshold=200,  # 200
                 filterByArea=True,
                 minArea=0.01, # 1500
                 filterByCircularity=True,
                 minCircularity=0.1,
                 filterByConvexity=True,
                 minConvexity=0.01, # 0.87
                 filterByInertia=True,
                 minInertiaRatio = 0.01)

BS_first_frame_to_process = 40
BS_last_frame_to_process = 205  # demo 205 frames stabilized 205 frames
combMask_until_this_frame = 0

# ~~~ Median ~~~ #
median_background_img = osp.join(cur_path, 'Temp', 'background_improved.jpg')
median_background50_img = osp.join(cur_path, 'Temp', 'background_improved50_50.jpg')
median_filter_frames_num = 10 #30 for DEMO. for our Stabilized
mask_max_diff_from_median =2
medianSaved = True
area_filter_parameter=10

# ~~~ Matting ~~~ #
MAT_frame_reduction_DEBUG = 0

forePlotted = False
backPlotted = False
plot_from_here = 50
plot_until_here = 52

tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'CSRT']
tracker_type = tracker_types[2]
bbox = (60, 200, 330, 815)  # Define an initial bounding box
epsilon = 0.00001
bwperim = dict(n=4)
MAT_dilate = dict(iterations=2, struct_size=(2, 2))  # wider 5,  (5,5)
r = 1  # 0<r<2   -can change to desired value
