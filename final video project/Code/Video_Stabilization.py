import cv2
import numpy as np
import config
import logging
from tqdm import tqdm

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
    T = cv2.getRotationMatrix2D((s[1] / 2, s[0] / 2), 0, 1.08) #was 1.04
    frame = cv2.warpAffine(frame, T, (s[1], s[0]))
    return frame

def Video_Stabilization():

    print("\nVideo_Stabilization:")
    # Read input video
    cap = cv2.VideoCapture(config.in_vid_file)

    # Get frame count
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Get width and height of video stream
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Get frames per second (fps)
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Define the codec for output video
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')

    # Set up output video
    out = cv2.VideoWriter(config.stabilized_vid_file, fourcc, fps, (w, h)) #'stabilize.avi'
    #out = cv2.VideoWriter('video_out.avi', fourcc, fps, (2 * w, h))

    # Read first frame
    _, prev = cap.read()

    # Convert frame to grayscale
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    # Pre-define transformation-store array
    transforms = np.zeros((n_frames - 1, 3), np.float32)

    print("\nprocess frames..")
    for i in tqdm(range(n_frames-1)):
        # Detect feature points in previous frame
        #prev_gray = cv2.bilateralFilter(prev_gray, 9, 250, 250)
        #prev_gray = cv2.Laplacian(prev_gray, cv2.CV_64F)
        #prev_gray = cv2.medianBlur(prev_gray, 21) very bad
        #prev_gray_x = cv2.Sobel(prev_gray, cv2.CV_8U, 1, 0)
        #prev_gray_y = cv2.Sobel(prev_gray, cv2.CV_8U, 0, 1)
        #prev_gray = 5*(prev_gray_x + prev_gray_y)
        #plt.figure()
        #plt.imshow(prev_gray)
        #plt.show()
        prev_pts = cv2.goodFeaturesToTrack(prev_gray,
                                           config.goodFeaturesToTrack['maxCorners'],  # was 200 #mine 1000
                                           config.goodFeaturesToTrack['qualityLevel'],  # was 0.01
                                           config.goodFeaturesToTrack['minDistance'],  #was 30
                                           config.goodFeaturesToTrack['blockSize'])  #was 3 #mine 5
        # Read next frame
        success, curr = cap.read()
        if not success:
            break

            # Convert to grayscale
        curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
        #curr_gray_x = cv2.Sobel(curr_gray, cv2.CV_8U, 1, 0)
        #curr_gray_y = cv2.Sobel(curr_gray, cv2.CV_8U, 0, 1)
        #curr_gray = 5*(curr_gray_x + curr_gray_y)

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

        # Extract traslation
        dx = m[0, 2]
        dy = m[1, 2]

        # Extract rotation angle
        da = np.arctan2(m[1, 0], m[0, 0])

        # Store transformation
        transforms[i] = [dx, dy, da]

        # Move to next frame
        prev_gray = curr_gray

        logging.debug("Frame: " + str(i) + "/" + str(n_frames) + " -  Tracked points : " + str(len(prev_pts)))
        #Or - why not print n_frames-1 ???

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
    print("\nWarrping image..")
    for i in tqdm(range(n_frames-1)):
        # Read next frame
        success, frame = cap.read()
        if not success:
            break

        # Extract transformations from the new transformation array
        dx = transforms_smooth[i, 0]
        dy = transforms_smooth[i, 1]
        da = transforms_smooth[i, 2]

        # Reconstruct transformation matrix accordingly to new values
        m = np.zeros((2, 3), np.float32)
        m[0, 0] = np.cos(da)
        m[0, 1] = -np.sin(da)
        m[1, 0] = np.sin(da)
        m[1, 1] = np.cos(da)
        m[0, 2] = dx
        m[1, 2] = dy

        # Apply affine wrapping to the given frame
        frame_stabilized = cv2.warpAffine(frame, m, (w, h))

        # Fix border artifacts
        frame_stabilized = fixBorder(frame_stabilized)

        # Write the frame to the file
        frame_out = frame_stabilized
        #frame_out = cv2.hconcat([frame_stabilized,frame_stabilized])
        #frame_out = cv2.hconcat([frame, frame_stabilized])

        # If the image is too big, resize it.
        #  if(frame_out.shape[1] > 1920):
        #    frame_out = cv2.resize(frame_out, (int(frame_out.shape[1]/2)), (int(frame_out.shape[0]/2)));
        cv2.putText(frame_out, "frame number : " + str(i), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

        # cv2.imshow("Before and After", frame_out)
        # cv2.waitKey(10)
        out.write(frame_out)
        if (i==n_frames-2):
            out.write(frame_out)
    # Release video
    cap.release()
    out.release()
    # Close windows
    cv2.destroyAllWindows()
    return
