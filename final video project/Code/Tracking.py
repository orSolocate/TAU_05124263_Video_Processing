import cv2
import config
import video_handling
import logging
from tqdm import tqdm


def Tracking():
    print("\nTracking Block:")
    # choose tracker
    tracker = video_handling.choose_tracker(tracker_type=config.output_tracker_type)
    # [input video]
    video = cv2.VideoCapture(config.matted_vid_file)
    n_frames, fourcc, fps, out_size = video_handling.extract_video_params(video)

    #  [output video]
    out = cv2.VideoWriter(config.out_vid_file, fourcc, fps, out_size)

    for i in tqdm(range(n_frames)):
        # Read a new frame
        ok, frame = video.read()
        if (i == 1):
            if not ok:
                logging.error('Unable to read video file')
                exit(0)
            tracker.init(frame, config.track_bbox)

        # Start timer
        timer = cv2.getTickCount()

        # Update tracker
        ok, bbox = tracker.update(frame)

        # Calculate Frames per second (FPS)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

        # Draw bounding box on frame
        if ok: # Tracking succeeded
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
		else: #Tracking failed
            cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
        # Display on frame - for debug
        cv2.putText(frame, "frame number : " + str(i), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)
        out.write(frame)
    return
