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

def fixBorder_inverse(frame):
    s = frame.shape
    T = cv2.getRotationMatrix2D((s[1] / 2, s[0] / 2), 0, 1.08)
    T = cv2.invertAffineTransform(T)
    frame = cv2.warpAffine(frame, T, (s[1], s[0]))
    return frame
