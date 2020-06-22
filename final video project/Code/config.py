import logging
import os.path as osp
import numpy as np

# ~~~ DEMO ~~~ #
DEMO = False  # True for given stabilized

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

outputs_vector=[stabilized_vid_file,extracted_vid_file,binary_vid_file,matted_vid_file,alpha_vid_file
                ,un_alpha_vid_file,out_vid_file]

# ~~~ logger ~~~ #
log_file = osp.join(cur_path, 'Outputs', 'RunTimeLog.txt')
logging.basicConfig(filename=log_file, filemode='w',
                    level=logging.DEBUG)  # no debug: level-logging.INFO, for debug: level=logging.DEBUG
logging.getLogger("matplotlib.legend").disabled = True
logging.getLogger("matplotlib.pyplot").disabled = True
logging.getLogger("matplotlib").disabled = True
logging.getLogger("matplotlib.font_manager").disabled = True

if DEMO:
    ##################Start:This are the parameters for supplied stabalized video!!!#######################
    # ~~~ Video_Stabilization ~~~ #
    SMOOTHING_RADIUS = 150  # was 50 #mine 75  The larger the more stable the video, but less reactive to sudden panning

    goodFeaturesToTrack = dict(maxCorners=1000,
                               qualityLevel=0.05,
                               minDistance=10,
                               blockSize=3)
    calcOpticalFlowPyrLK = dict(winSize=(50, 50),
                                maxLevel=10)
    transforms_file = osp.join(cur_path, 'Temp', 'video_stabilization_transforms.p')

    # ~~~ Background_Substraction ~~~ #
    BS_frame_reduction_DEBUG = 0
    createBackground_Substraction = dict(history=170,  # demo 170
                                         dist2Threshold=500.0,  # demo 500.0
                                         detectShadows=True)
    backSub_apply = dict(learningRate=-1)  # negative means automatic learningRate #demo -1
    BS_erode = dict(iterations=2, struct_size=(15, 10))  # thiner 1, (20,10) #demo 2, (15.10)
    BS_dilate = dict(iterations=5, struct_size=(4, 4))  # wider 5,  (5,5) #demo 5, (4,$)
    blob_detect = dict(minThreshold=0,
                       maxThreshold=200,
                       filterByArea=True,
                       minArea=0.01,
                       filterByCircularity=True,
                       minCircularity=0.1,
                       filterByConvexity=True,
                       minConvexity=0.01,
                       filterByInertia=True,
                       minInertiaRatio=0.01)

    BS_first_frame_to_process = 0
    BS_last_frame_to_process = 205  # demo 205 frames stabilized 205 frames
    combMask_until_this_frame = 28
    comb_shorts = dict(lower_bound=np.array([0, 0, 0]),  # demo [0,0,0]
                       upper_bound=np.array([50, 50, 50]),  # demo [50,50,50,]
                       er_struct_size=(10, 5),  # demo (10,5), wider: (15,10)
                       er_iterations=2,  # demo 2
                       dil_struct_size=(5, 5),  # demo (5,5)
                       dil_iterations=5)  # demo 5

    comb_shirt = dict(lower_bound=np.array([0, 0, 55]),  # demo [0,0,55] [0,0,40]
                      upper_bound=np.array([90, 80, 124]),  # demo [90,80,124]
                      er_struct_size=(15, 10),  # demo 15,10
                      er_iterations=2,  # demo 2
                      dil_struct_size=(5, 5),  # demo 5,5
                      dil_iterations=5)  # demo 5

    comb_legs = dict(lower_bound=np.array([60, 75, 128]),  # demo [60,75,124]
                     upper_bound=np.array([95, 110, 160]),  # demo [95,110,160]
                     er_struct_size=(5, 5),  # demo (5,5)
                     er_iterations=2,  # demo 2
                     dil_struct_size=(5, 5),  # demo (5,5)
                     dil_iterations=5)  # demo 5

    comb_shoes = dict(lower_bound=np.array([0, 0, 0]),  # demo [0,0,0]
                      upper_bound=np.array([95, 98, 100]),  # demo [95,98,100]
                      er_struct_size=(10, 10),  # demo (10,10)
                      er_iterations=2,  # demo 2
                      dil_struct_size=(5, 5),  # demo (5,5)
                      dil_iterations=5)  # demo 5

    comb_skin = dict(lower_bound=np.array([75, 85, 140]),  # demo [75,85,140]
                     upper_bound=np.array([110, 120, 180]),  # demo [110,120,180]
                     er_struct_size=(5, 5),  # demo (5,5) ,  wider:(15,10)
                     er_iterations=2,  # demo 2
                     dil_struct_size=(5, 5),  # demo (5,5)
                     dil_iterations=5)  # demo 5

    # ~~~ Median ~~~ #
    median_background_img = osp.join(cur_path, 'Temp', 'background.jpg')
    median_background50_img = osp.join(cur_path, 'Temp', 'background50_50.jpg')
    median_filter_frames_num = 30  # 30 for DEMO. 10 for our Stabilized
    mask_max_diff_from_median = 3  # demo 3 stabilzed 25 (not so good)
    medianSaved = True
    area_filter_parameter = 10

    # ~~~ Matting ~~~ #
    MAT_frame_reduction_DEBUG = 0

    forePlotted = False
    backPlotted = False
    plot_from_here = 0
    plot_until_here = 0

    tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'CSRT']
    tracker_type = tracker_types[6]
    bbox = (0, 200, 330, 815)  # (demo) (0, 200, 330, 815) Define an initial bounding box
    epsilon = 0.00001
    bwperim = dict(n=4)
    MAT_dilate = dict(iterations=2, struct_size=(2, 2))  # wider 5,  (5,5)
    r = 0.0001  # 0<r<2

    # ~~~ Tracking ~~~ #
    track_bbox = (60, 200, 330, 815)  # demo(60, 200, 330, 815)
######################End:This are the parameters for supplied stabalized video!!!#####################
else:
    #######################Start: This are the parameters for our stabalized video!!!######################
    # ~~~ Video_Stabilization ~~~ #
    SMOOTHING_RADIUS = 150  # was 50 #mine 75  The larger the more stable the video, but less reactive to sudden panning

    goodFeaturesToTrack = dict(maxCorners=1000,  # was 200 #mine 1000
                               qualityLevel=0.05,  # was 0.01
                               minDistance=10,  # was 30
                               blockSize=3)  # was 3 #mine 5
    calcOpticalFlowPyrLK = dict(winSize=(50, 50),
                                maxLevel=10)
    transforms_file = osp.join(cur_path, 'Temp', 'video_stabilization_transforms.p')

    # ~~~ Background_Substraction ~~~ #
    BS_frame_reduction_DEBUG = 0
    createBackground_Substraction = dict(history=170,  # demo 170
                                         dist2Threshold=500.0,  # 800 demo 500.0
                                         detectShadows=True)
    backSub_apply = dict(learningRate=-1)  # negative means automatic learningRate #demo -1
    BS_erode = dict(iterations=2, struct_size=(20, 15))  # (20,15) thiner 1, #demo 2, (15.10)
    BS_dilate = dict(iterations=5, struct_size=(4, 4))  # wider 5,  (5,5) #demo 5, (4,$)
    blob_detect = dict(minThreshold=0,  # 10
                       maxThreshold=200,  # 200
                       filterByArea=True,
                       minArea=0.01,  # 1500
                       filterByCircularity=True,
                       minCircularity=0.1,
                       filterByConvexity=True,
                       minConvexity=0.01,  # 0.87
                       filterByInertia=True,
                       minInertiaRatio=0.01)

    BS_first_frame_to_process = 0
    BS_last_frame_to_process = 205  # demo 205 frames stabilized 205 frames
    combMask_until_this_frame = 28
    comb_shorts = dict(lower_bound=np.array([0, 0, 0]),  # demo [0,0,0]
                       upper_bound=np.array([50, 50, 50]),  # demo [50,50,50,]
                       er_struct_size=(10, 5),  # demo (10,5), wider: (15,10)
                       er_iterations=2,  # demo 2
                       dil_struct_size=(5, 5),  # demo (5,5)
                       dil_iterations=5)  # demo 5

    comb_shirt = dict(lower_bound=np.array([40, 40, 40]),  # demo [0,0,55]   #######creating noise!!!
                      upper_bound=np.array([90, 80, 130]),  # [90, 80, 130]
                      # demo [90,80,124]  stabilized improve1 [83,80,115] improve 2
                      er_struct_size=(15, 10),  # demo 15,10
                      er_iterations=2,  # demo 2
                      dil_struct_size=(5, 5),  # demo 5,5
                      dil_iterations=5)  # demo 5

    comb_legs = dict(lower_bound=np.array([70, 85, 130]),  # [60, 75, 128]     demo [60,75,124]
                     upper_bound=np.array([95, 110, 160]),  # demo [95,110,160]
                     er_struct_size=(5, 5),  # demo (5,5)
                     er_iterations=2,  # demo 2
                     dil_struct_size=(5, 5),  # demo (5,5)
                     dil_iterations=5)  # demo 5

    comb_shoes = dict(lower_bound=np.array([0, 0, 0]),  # demo [0,0,0]
                      upper_bound=np.array([95, 98, 100]),  # demo [95,98,100]
                      er_struct_size=(10, 10),  # demo (10,10)
                      er_iterations=2,  # demo 2
                      dil_struct_size=(5, 5),  # demo (5,5)
                      dil_iterations=5)  # demo 5

    comb_skin = dict(lower_bound=np.array([85, 90, 140]),  # best[85,85,140]   demo [75,85,140]
                     upper_bound=np.array([100, 110, 170]),  # [110, 120, 180] demo [110,120,180]
                     er_struct_size=(5, 5),  # demo (5,5) ,  wider:(15,10)
                     er_iterations=2,  # demo 2
                     dil_struct_size=(5, 5),  # demo (5,5)
                     dil_iterations=5)  # demo 5

    # ~~~ Median ~~~ #
    median_background_img = osp.join(cur_path, 'Temp', 'background.jpg')
    median_background50_img = osp.join(cur_path, 'Temp', 'background50_50.jpg')
    median_filter_frames_num = 10  # 30 for DEMO. 10 for our Stabilized
    mask_max_diff_from_median = 25  # demo 3 stabilzed 25 (not so good)
    medianSaved = True
    area_filter_parameter = 10

    # ~~~ Matting ~~~ #
    MAT_frame_reduction_DEBUG = 0

    forePlotted = False
    backPlotted = False
    plot_from_here = 0
    plot_until_here = 0

    tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'CSRT']
    tracker_type = tracker_types[6]
    bbox = (0, 200, 330, 815)  # (demo) (0, 200, 330, 815) Define an initial bounding box
    epsilon = 0.00001
    bwperim = dict(n=4)
    MAT_dilate = dict(iterations=2, struct_size=(2, 2))  # wider 5,  (5,5)
    r = 0.0001  # 0<r<2

    # ~~~ Tracking ~~~ #
    track_bbox = (60, 200, 330, 815)  # demo(60, 200, 330, 815)
    #######################End: This are the parameters for our stabalized video!!!#####################
