import cv2
import numpy as np
import config
from tqdm import tqdm
import video_handling
import pickle
import time


def movingAverage(curve, radius):
    window_size = 2 * radius + 1
    # Define the filter
    f = np.ones(window_size) / window_size
    # Add padding to the boundaries
    curve_pad = np.lib.pad(curve, (radius, radius), 'edge')
    # Apply convolution
    curve_smoothed = np.convolve(curve_pad, f, mode='same')
    # Remove padding
    curve_smoothed = curve_smoothed[radius:-radius]
    # return smoothed curve
    return curve_smoothed


def smooth(trajectory):
    smoothed_trajectory = np.copy(trajectory)
    # Filter the x, y and angle curves
    for i in range(3):
        smoothed_trajectory[:, i] = movingAverage(trajectory[:, i], radius=config.SMOOTHING_RADIUS)

    return smoothed_trajectory


def fixBorder(frame):
    s = frame.shape
    # Scale the image 4% without moving the center
    T = cv2.getRotationMatrix2D((s[1] / 2, s[0] / 2), 0, 1.08)  # was 1.04
    frame = cv2.warpAffine(frame, T, (s[1], s[0]))
    return frame


def Video_Stabilization():
    print("\n\Video_Stabilization Block:\n")
    # Read input video
    cap = cv2.VideoCapture(config.in_vid_file)
    n_frames, fourcc, fps, out_size = video_handling.extract_video_params(cap)

    out = cv2.VideoWriter(config.stabilized_vid_file, fourcc, fps, out_size)

    # Read first frame
    _, prev = cap.read()

    # Convert frame to grayscale
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    # Pre-define transformation-store array
    transforms = np.zeros((n_frames - 1, 3), np.float32)

    print("calculating stabilizing transforms..")
    for i in tqdm(range(n_frames - 1)):
        prev_pts = cv2.goodFeaturesToTrack(prev_gray,
                                           config.goodFeaturesToTrack['maxCorners'],
                                           config.goodFeaturesToTrack['qualityLevel'],
                                           config.goodFeaturesToTrack['minDistance'],
                                           config.goodFeaturesToTrack['blockSize'])
        # Read next frame
        success, curr = cap.read()
        if not success:
            break

        # Convert to grayscale
        curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
        # Calculate optical flow (i.e. track feature points)
        curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None,
                                                         config.calcOpticalFlowPyrLK['winSize'],
                                                         config.calcOpticalFlowPyrLK['maxLevel'])
        # Sanity check
        assert prev_pts.shape == curr_pts.shape
        # Filter only valid points
        idx = np.where(status == 1)[0]
        prev_pts = prev_pts[idx]
        curr_pts = curr_pts[idx]
        # Find transformation matrix
        m = cv2.estimateRigidTransform(prev_pts, curr_pts, fullAffine=True)  # will only work with OpenCV-3 or less
        # Extract translation
        dx = m[0, 2]
        dy = m[1, 2]

        # Extract rotation angle
        da = np.arctan2(m[1, 0], m[0, 0])

        # Store transformation
        transforms[i] = [dx, dy, da]

        # Move to next frame
        prev_gray = curr_gray

    # Compute trajectory using cumulative sum of transformations
    trajectory = np.cumsum(transforms, axis=0)

    # Create variable to store smoothed trajectory
    smoothed_trajectory = smooth(trajectory)

    # Calculate difference in smoothed_trajectory and trajectory
    difference = smoothed_trajectory - trajectory

    # Calculate newer transformation array
    transforms_smooth = transforms + difference

    # Reset stream to first frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Write n_frames-1 transformed frames
    print("Warping image..")
    time.sleep(0.5)
    for i in tqdm(range(n_frames - 1)):
        # Read next frame
        success, frame = cap.read()
        # sanity check
        if not success:
            break

        m=video_handling.prepare_wrap_transform(transforms_smooth[i,:])
        # Apply affine wrapping to the given frame
        frame_stabilized = cv2.warpAffine(frame, m, out_size)

        # Fix border artifacts
        frame_stabilized = fixBorder(frame_stabilized)

        # Write the frame to the file
        frame_out = frame_stabilized
        out.write(frame_out)
        if (i == n_frames - 2):
            out.write(frame_out)
    # Release video
    cap.release()
    out.release()
    # Close windows
    cv2.destroyAllWindows()

    pickle.dump(transforms_smooth, open(config.transforms_file, "wb"))
    return
