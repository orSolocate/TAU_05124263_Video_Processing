import cv2
from unused.compute_distance_map import *
import config
from tqdm import tqdm
import logging
import video_handling
import pickle
import Matting_functions



def Matting():
    print("\nMatting Block:")
    #input videos
    if (config.DEMO):
        capture_stab = cv2.VideoCapture(config.demo_stabilized_vid_file)
    else:
        capture_stab = cv2.VideoCapture(config.stabilized_vid_file)
    n_frames_stab,fourcc, fps, out_size=video_handling.extract_video_params(capture_stab)
    capture_bin = cv2.VideoCapture(config.binary_vid_file)
    n_frames_bin, fourcc_bin, fps_bin, out_size_bin = video_handling.extract_video_params(capture_bin)

    #output videos
    out = cv2.VideoWriter(config.matted_vid_file, fourcc, fps,out_size)
    alpha_out = cv2.VideoWriter(config.alpha_vid_file, fourcc, fps, out_size)
    un_alpha_out = cv2.VideoWriter(config.un_alpha_vid_file, fourcc, fps, out_size)

    alpha_list=[]
    #read transformations list for unstabilized_alpha
    transforms_smooth = pickle.load(open(config.transforms_file, "rb"))

    # read background image
    background = cv2.imread(config.in_background_file)
    background_hsv = cv2.cvtColor(background, cv2.COLOR_BGR2HSV)

    # Start timer
    timer = cv2.getTickCount()

    #choose tracker
    tracker = video_handling.choose_tracker(tracker_type=config.matting_tracker_type)

    print("\nprocessing frames for Matting..")
    forePlotted = config.forePlotted
    backPlotted = config.backPlotted
    for iteration in tqdm(range(0,min(n_frames_stab,n_frames_bin)-config.MAT_frame_reduction_DEBUG)):
        ret, frame_stab = capture_stab.read()
        ret, frame_bin = capture_bin.read()
        frame_stab_hsv = cv2.cvtColor(frame_stab, cv2.COLOR_BGR2HSV)  # 29=scrib, 255=garbage

        gray_fore_scrib = cv2.cvtColor(frame_bin, cv2.COLOR_BGR2GRAY)  # 29=scrib, 255=garbage
        gray_fore_scrib[gray_fore_scrib>=200]=100
        gray_fore_scrib[gray_fore_scrib<=99]=0
        gray_back_scrib = cv2.bitwise_not(cv2.cvtColor(frame_bin, cv2.COLOR_BGR2GRAY))
        gray_back_scrib[gray_back_scrib<=99]=0
        gray_back_scrib[gray_back_scrib>=200]=100

        #  [tracker]
        # Initialize tracker with first frame and bounding box
        ok = tracker.init(frame_stab, config.bbox)
        # Update tracker
        ok, config.bbox = tracker.update(frame_stab)

        # Calculate Frames per second (FPS)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);

        # Display FPS on frame
        cv2.putText(frame_stab, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)


        frame_stab_tracked = frame_stab[int(config.bbox[1]):int((config.bbox[1] + config.bbox[3])),int(config.bbox[0]):int(1.3*(config.bbox[0] + config.bbox[2])),:]
        gray_back_scrib = gray_back_scrib[int(config.bbox[1]):int((config.bbox[1] + config.bbox[3])),int(config.bbox[0]):int(1.3*(config.bbox[0] + config.bbox[2]))]
        gray_fore_scrib = gray_fore_scrib[int(config.bbox[1]):int((config.bbox[1] + config.bbox[3])),int(config.bbox[0]):int(1.3*(config.bbox[0] + config.bbox[2]))]

        frame_stab_hsv_tracked = cv2.cvtColor(frame_stab_tracked, cv2.COLOR_BGR2HSV)
        frame_stab_tracked = frame_stab_hsv_tracked[:,:,2]

        #takeing only points from scribble
        background_array = frame_stab_tracked[gray_back_scrib == 100]
        foreground_array = frame_stab_tracked[gray_fore_scrib == 100]

        #binary maske normalized to [0,1]
        frame_bin = frame_bin/255

        #calculate KDE for background and foreground
        kde_fore_pdf=Matting_functions.kde_evaluate(foreground_array,forePlotted,title='Kernel Density Estimation - Background')
        kde_back_pdf=Matting_functions.kde_evaluate(background_array,backPlotted,title='Kernel Density Estimation - Foreground')

        #probabilties of background and foreground
        P_F = kde_fore_pdf / (kde_fore_pdf + kde_back_pdf)
        P_B = kde_back_pdf / (kde_fore_pdf + kde_back_pdf)

        #probabilies map of background and foreground
        Probability_map_fore = P_F[frame_stab_tracked]
        Probability_map_back = P_B[frame_stab_tracked]

        gaodesic_fore=Matting_functions.geo_distance(frame_stab_tracked, gray_fore_scrib, forePlotted)
        gaodesic_back=Matting_functions.geo_distance(frame_stab_tracked, gray_back_scrib, backPlotted)

        if (config.plot_from_here<iteration<=config.plot_until_here):
            forePlotted=True
            backPlotted=True
        else:
            forePlotted = config.forePlotted
            backPlotted = config.backPlotted

        foreground_tracked = np.zeros((frame_stab_tracked.shape))
        foreground_tracked[gaodesic_fore-gaodesic_back<=0] = frame_stab_tracked[gaodesic_fore-gaodesic_back<=0]

        #make binary map from foreground
        Vforeground = frame_stab_tracked
        Vforeground = (gaodesic_fore - gaodesic_back >= 0) * np.zeros(Vforeground.shape)
        Vforeground = (gaodesic_fore - gaodesic_back < 0) * np.ones(Vforeground.shape)

        #extracting foreground outline
        delta_edges = Matting_functions.bwperim(Vforeground, n=config.bwperim['n'])

        #Morphological dilate
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, config.MAT_dilate['struct_size'])
        B_ro = cv2.dilate(delta_edges, kernel, iterations=config.MAT_dilate['iterations'])  # wider binary image
        B_ro = (B_ro==1) * 0.5

        trimap = np.zeros((Vforeground.shape))
        trimap = Vforeground#foreground area filled with '1'
        trimap[B_ro == 0.5] =  0.5# undecided area

        #alpha mating
        r = config.r * np.ones((frame_stab_tracked.shape))
        gaodesic_fore[gaodesic_fore==0] = config.epsilon # prevent dividing by zero
        gaodesic_back[gaodesic_back == 0] = config.epsilon
        w_fore = cv2.multiply(np.power(gaodesic_fore,-r),Probability_map_fore)
        w_back = cv2.multiply(np.power(gaodesic_back,-r),Probability_map_back)

        alpha = np.zeros((frame_stab_tracked.shape))
        alpha[trimap == 0.5] = w_fore[trimap == 0.5] / (w_fore[trimap == 0.5] + w_back[trimap == 0.5]) #blending
        alpha[trimap == 0  ] = 0 #background
        alpha[trimap == 1  ] = 1 #foreground


        # Perform alpha blending
        mask = foreground_tracked>0
        foreground_mask_tracked= mask * np.ones(mask.shape)
        foreground = np.zeros(frame_stab_hsv.shape)
        foreground[int(config.bbox[1]): int(config.bbox[1] + config.bbox[3]), int(config.bbox[0]): int(1.3 * (config.bbox[0] + config.bbox[2])),0] =  foreground_mask_tracked * frame_stab_hsv[int(config.bbox[1]): int(config.bbox[1] + config.bbox[3]), int(config.bbox[0]): int(1.3 * (config.bbox[0] + config.bbox[2])),0]
        foreground[int(config.bbox[1]): int(config.bbox[1] + config.bbox[3]), int(config.bbox[0]): int(1.3 * (config.bbox[0] + config.bbox[2])),1] =  foreground_mask_tracked * frame_stab_hsv[int(config.bbox[1]): int(config.bbox[1] + config.bbox[3]), int(config.bbox[0]): int(1.3 * (config.bbox[0] + config.bbox[2])),1]
        foreground[int(config.bbox[1]): int(config.bbox[1] + config.bbox[3]), int(config.bbox[0]): int(1.3 * (config.bbox[0] + config.bbox[2])),2] =  foreground_tracked
        background = background_hsv

        alpha_full = np.zeros((background_hsv.shape))
        alpha_full[int(config.bbox[1]): int(config.bbox[1] + config.bbox[3]), int(config.bbox[0]): int(1.3 * (config.bbox[0] + config.bbox[2])),0] =  alpha
        alpha_full[int(config.bbox[1]): int(config.bbox[1] + config.bbox[3]), int(config.bbox[0]): int(1.3 * (config.bbox[0] + config.bbox[2])),1] =  alpha
        alpha_full[int(config.bbox[1]): int(config.bbox[1] + config.bbox[3]), int(config.bbox[0]): int(1.3 * (config.bbox[0] + config.bbox[2])),2] =  alpha

        alpha_full_out = np.uint8(alpha_full * 255)
        alpha_out.write(alpha_full_out)
        alpha_list.append(alpha_full_out)

        alpha_full=alpha_full.astype(float)
        foreground_mul_alpha = cv2.multiply(alpha_full, foreground)

        x=np.ones((alpha_full.shape)) - alpha_full
        y=background
        background_mul= cv2.multiply(x.astype(float) , y.astype(float))

        background_mul = background_mul.astype(int)
        foreground_mul_alpha = foreground_mul_alpha.astype(int)

        outImage = cv2.add(background_mul, foreground_mul_alpha)
        outImage = np.uint8(outImage)
        outImage = cv2.cvtColor(outImage, cv2.COLOR_HSV2BGR)

        # write output matted video
        out.write(outImage)
        #cv2.imshow('Matted', outImage)

    print("\nApplying inverse transform to write 'unestablized_alpha.avi'..")
    n_frames_alpha=len(alpha_list)
    for i in tqdm(range(n_frames_alpha-1)):
            # 'unfix' border artifacts
            un_alpha = Matting_functions.fixBorder_inverse(alpha_list[i])
            m=video_handling.prepare_wrap_transform(transforms_smooth[i, :])
            m=cv2.invertAffineTransform(m)
            un_alpha = cv2.warpAffine(un_alpha, m, out_size)
            # Write the frame to the file
            un_alpha_out.write(un_alpha)
            if (i == n_frames_alpha - 2):
                un_alpha_out.write(un_alpha) #write last frame twice
    # Release video
    cv2.destroyAllWindows()
    capture_stab.release()
    capture_bin.release()
    out.release()
    alpha_out.release()
    un_alpha_out.release()
    return
