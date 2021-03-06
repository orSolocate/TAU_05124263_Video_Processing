import cv2
import logging
import numpy as np

def extract_video_params(cap):
    #sanity checks for video
    if not cap.isOpened:
        logging.error('Unable to open video file:')
        exit(0)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # Define video codec
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_size = (width, height)

    return n_frames,fourcc, fps, out_size

def choose_tracker(tracker_type):
    if tracker_type == 'BOOSTING':
        return cv2.TrackerBoosting_create()
    if tracker_type == 'MIL':
        return cv2.TrackerMIL_create()
    if tracker_type == 'KCF':
        return  cv2.TrackerKCF_create()
    if tracker_type == 'TLD':
        return  cv2.TrackerTLD_create()
    if tracker_type == 'MEDIANFLOW':
        return  cv2.TrackerMedianFlow_create()
    if tracker_type == 'GOTURN':
        return cv2.TrackerGOTURN_create()
    if tracker_type == "CSRT":
        return cv2.TrackerCSRT_create()
    logging.error("Undefined tracker selected, pleasse choose another tracker from config.py")
    exit(1)
    return


def prepare_wrap_transform(transforms_smooth):
        # Extract transformations from the new transformation array
        dx = transforms_smooth[0]
        dy = transforms_smooth[1]
        da = transforms_smooth[2]

        # Reconstruct transformation matrix accordingly to new values
        m = np.zeros((2, 3), np.float32)
        m[0, 0] = np.cos(da)
        m[0, 1] = -np.sin(da)
        m[1, 0] = np.sin(da)
        m[1, 1] = np.cos(da)
        m[0, 2] = dx
        m[1, 2] = dy
        return m


def test_output(input_name,output_name):
    in_vid = cv2.VideoCapture(input_name)
    out_vid = cv2.VideoCapture(output_name)
    n_frames, fourcc, fps, out_size=extract_video_params(in_vid)
    n_frames_out, fourcc_out, fps_out, out_size_out=extract_video_params(out_vid)
    if (n_frames==n_frames_out and fourcc==fourcc and fps==fps_out and out_size==out_size_out):
        return True
    else:
        logging.error('Block output has different properties as input\n video path'+output_name+'\n')
        return False

def test_all_outputs(input_name, outputs_list):
    result=[]
    for output_name in outputs_list:
        result.append(test_output(input_name,output_name))
    result=np.asarray(result)
    if (np.all(result)==True):
        print('All output tests passed')
        return True
    return False

