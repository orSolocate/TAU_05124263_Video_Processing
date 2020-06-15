import logging
import os.path as osp

# ~~~ global ~~~ #
cur_path = osp.split(osp.dirname(osp.abspath(__file__)))[0]
log_file=osp.join(cur_path,'Outputs','RunTimeLog.txt')
logging.basicConfig(filename=log_file, filemode='w', level=logging.INFO) #for debug: logging.DEBUG

# ~~~ runme ~~~ #
in_vid_file = osp.join(cur_path, 'Input', 'INPUT.avi')
stabilized_vid_file = osp.join(cur_path, 'Outputs', 'stabilize.avi')
demo_stabilized_vid_file = osp.join(cur_path, 'Outputs', 'Stabilized_Example_INPUT.avi')
extracted_vid_file = osp.join(cur_path, 'Outputs', 'extracted.avi')
binary_vid_file = osp.join(cur_path, 'Outputs', 'binary.avi')
in_background_file = osp.join(cur_path, 'Input', 'background.jpeg')


# ~~~ Video_Stabilization ~~~ #
SMOOTHING_RADIUS = 150  # was 50 #mine 75  The larger the more stable the video, but less reactive to sudden panning

goodFeaturesToTrack = dict(maxCorners=1000,  # was 200 #mine 1000
                           qualityLevel=0.05,  # was 0.01
                           minDistance=10,  # was 30
                           blockSize=3)
calcOpticalFlowPyrLK = dict(winSize=(50, 50),
                            maxLevel=10)

# ~~~ Background_Substraction ~~~ #
createBackground_Substraction=dict(history = 120,
                             dist2Threshold = 400.0,
                             detectShadows = True)