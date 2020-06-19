import cv2
import config
import video_handling
import logging
from tqdm import tqdm

(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

def Tracking():
    tracker_type = config.tracker_type

    if int(minor_ver) < 3:
        tracker = cv2.Tracker_create(tracker_type)
    else:
        if tracker_type == 'BOOSTING':
            tracker = cv2.TrackerBoosting_create()
        if tracker_type == 'MIL':
            tracker = cv2.TrackerMIL_create()
        if tracker_type == 'KCF':
            tracker = cv2.TrackerKCF_create()
        if tracker_type == 'TLD':
            tracker = cv2.TrackerTLD_create()
        if tracker_type == 'MEDIANFLOW':
            tracker = cv2.TrackerMedianFlow_create()
        if tracker_type == 'GOTURN':
            tracker = cv2.TrackerGOTURN_create()
        if tracker_type == "CSRT":
            tracker = cv2.TrackerCSRT_create()

    # Read video
    video = cv2.VideoCapture(config.matted_vid_file)
    #video = cv2.VideoCapture(config.stabilized_vid_file) #for debug
    n_frames,fourcc, fps, out_size=video_handling.extract_video_params(video)

    out = cv2.VideoWriter(config.out_vid_file, fourcc, fps, out_size)
    # Uncomment the line below to select a different bounding box
    #bbox = cv2.selectROI(frame, False)

    for i in tqdm(range(n_frames)): #-1 because we already read the first frame
        # Read a new frame
        ok, frame = video.read()
        if (i==1):
            if not ok:
                logging.error('Unable to read video file')
                exit(0)
            ok = tracker.init(frame, config.bbox)

        # Start timer
        timer = cv2.getTickCount()

        # Update tracker
        ok, bbox = tracker.update(frame)

        # Calculate Frames per second (FPS)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);

        # Draw bounding box
        if ok:
            # Tracking success
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
        else:
            # Tracking failure
            cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        # Display on frame
        cv2.putText(frame, tracker_type + " Tracker", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);
        cv2.putText(frame, "frame number : " + str(i), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)
        cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);

        # Display result -debug
        #cv2.imshow("Tracking", frame)
        # k = cv2.waitKey(1) & 0xff# Exit if ESC pressed
        # if k == 27: break
        out.write(frame)
    return
