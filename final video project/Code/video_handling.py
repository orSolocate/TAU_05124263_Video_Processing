import cv2
import logging

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
