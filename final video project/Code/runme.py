import config, Video_Stabilization, Background_Substraction, Matting_Tracker_binary, Tracking, video_handling
import timeit
import logging
import os


def runtime_update(step_name, start):
    elapsed_time = timeit.default_timer() - start
    logging.info("%s=   %f seconds", step_name, elapsed_time)
    return elapsed_time


def runme():
    # Initializing step
    start = timeit.default_timer()
    Total_time = 0
    if not os.path.exists(os.path.join(config.cur_path, 'Outputs')):
        os.makedirs(os.path.join(config.cur_path, 'Outputs'))
    Total_time += runtime_update("Initalizing runtime", start)
    # Video_Stabilization step
    start = timeit.default_timer()
    Video_Stabilization.Video_Stabilization()
    Total_time += runtime_update("Video_Stabilization runtime", start)
    # Background_Subtraction step
    start = timeit.default_timer()
    Background_Substraction.Background_Substraction()
    Total_time += runtime_update("Background_Subtraction runtime", start)
    # Matting step
    start = timeit.default_timer()
    Matting_Tracker_binary.Matting()
    Total_time += runtime_update("Matting runtime", start)
    # Tracking step
    start = timeit.default_timer()
    Tracking.Tracking()
    Total_time += runtime_update("Tracking runtime", start)
    logging.info("Total Execution runtime=  %f seconds", Total_time)
    # test
    video_handling.test_all_outputs(config.in_vid_file, config.outputs_vector)
    # close logger
    logging.shutdown()
    return


if __name__ == '__main__':
    runme()
