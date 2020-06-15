from Code import config, Video_Stabilization, Background_Substraction
import timeit
import logging
import os


def runtime_update(step_name, start):
    elapsed_time = timeit.default_timer() - start
    logging.info("%s=   %f seconds", step_name, elapsed_time)
    return elapsed_time


def runme():
    # initalizing step
    start = timeit.default_timer()
    Total_time = 0
    if not os.path.exists(os.path.join(config.cur_path, 'Outputs')):
        os.makedirs(os.path.join(config.cur_path, 'Outputs'))
    Total_time += runtime_update("Initalizing runtime", start)
    # elapsed_time = timeit.default_timer() - start
    # elapsed_time
    # logging.info("Initalizing runtime=  %f seconds", elapsed_time)

    # Video_Stabilization step
    start = timeit.default_timer()
    Video_Stabilization.Video_Stabilization()
    Total_time += runtime_update("Video_Stabilization runtime", start)
    # elapsed_time = timeit.default_timer() - start
    # Total_time += elapsed_time
    # logging.info("Video_Stabilization runtime=  %f seconds", elapsed_time)

    # Background_Substraction step
    start = timeit.default_timer()
    Background_Substraction.Background_Substraction()
    Total_time += runtime_update("Background_Substraction runtime", start)
    # elapsed_time = timeit.default_timer() - start
    # Total_time += elapsed_time
    # logging.info("Background_Substraction runtime=  %f seconds", elapsed_time)

    # Matting step
    start = timeit.default_timer()
    # Matting.Matting()
    Total_time += runtime_update("Matting runtime", start)
    # elapsed_time = timeit.default_timer() - start
    # Total_time += elapsed_time
    # logging.info("Matting runtime=  %f seconds", elapsed_time)

    # Tracking step
    start = timeit.default_timer()
    # Tracking.Tracking()
    Total_time += runtime_update("Tracking runtime", start)
    # elapsed_time = timeit.default_timer() - start
    # Total_time += elapsed_time
    # logging.info("Tracking runtime=  %f seconds", elapsed_time)

    # close all
    logging.info("Total Execution runtime=  %f seconds", Total_time)
    logging.shutdown()
    return


if __name__ == '__main__':
    runme()
