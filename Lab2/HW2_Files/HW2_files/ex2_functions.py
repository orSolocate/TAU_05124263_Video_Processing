import numpy as np
import cv2
from scipy import signal
from scipy.interpolate import griddata

# ID1 = 204356315
# ID2 = 200846038

def  LucasKanadeStep(I1,I2,WindowSize):
    Threshold = 0.00001
    #matrix used for derivative calculations
    ones_matrix = np.ones((WindowSize,WindowSize))
    mode = 'same'

    # for each point, calculate Ix, Iy, It
    It = I2 - I1
    I2x = cv2.Sobel(I2, cv2.CV_64F, 1, 0, ksize=3)# x axis derivatie
    I2y = cv2.Sobel(I2, cv2.CV_64F, 0, 1, ksize=3)# y axis derivatie
    I2xx = signal.convolve2d(I2x * I2x, ones_matrix, mode)
    I2yy = signal.convolve2d(I2y * I2y, ones_matrix, mode)
    I2xy = signal.convolve2d(I2x * I2y, ones_matrix, mode)
    Ixt = signal.convolve2d(I2x * It, ones_matrix, mode)
    Iyt = signal.convolve2d(I2y * It, ones_matrix, mode)

    #matrix determinate calculation for none inversed
    det = I2xx * I2yy - I2xy * I2xy
    det_coeff = np.zeros(np.shape(det))
    det_mask = det > Threshold
    det_coeff[det_mask] = 1/det[det_mask] #mask for small numbers - faster calculation

    #delta p matrix implemantion for whole photo
    Matrix_mul_row_1 = I2yy * Ixt - I2xy * Iyt #based on lab 2 slide 20
    Matrix_mul_row_2 = -I2yy * Ixt + I2xx * Iyt #based on lab 2 slide 20
    du = -det_coeff * (Matrix_mul_row_1)
    dv = -det_coeff * (Matrix_mul_row_2)
    return du,dv


def WarpImage(I,u,v):
    #make a 3D grid in the size of I input photo
    M = I.shape[0] #rows
    N = I.shape[1] #columns

    x_axis = np.arange(0,N)
    y_axis = np.arange(0,M)
    #creates matrices for x values and y values
    x,y = np.meshgrid(x_axis,y_axis)

    #represent data grid and pictures as flat arrays
    I_array = np.reshape(I, (M*N,1))
    x_array = np.reshape(x, (M*N,1))
    y_array = np.reshape(y, (M*N,1))

    #built an (x,y) data grid
    cordinate = np.zeros((M*N,2))
    cordinate[:,0] = x_array[:,0]
    cordinate[:,1] = y_array[:,0]

    #now we change the grid to a float type and go back to the original cordinate
    x = x.astype(float)
    x = x + u
    y = y.astype(float)
    y = y + v

    #interpolation procedure
    pic_after_interpolation = griddata(cordinate, I_array[:,0], (x,y), method = 'linear')
    #use original data out of bounds
    pic_after_interpolation[(x > N-1)] = I[(x > N-1)]
    pic_after_interpolation[(y > M-1)] = I[(y > M-1)]
    pic_after_interpolation[(x < 0)] = I[(x < 0)]
    pic_after_interpolation[(y < 0)] = I[(y < 0)]
    return pic_after_interpolation


def LucasKanadeOpticalFlow(I1,I2,WindowSize,Maxlter,NumLevels):
    #assumes NumLevels >=1
    I1_pyramids=[I1]
    I2_pyramids=[I2]

    for i in range(1,NumLevels):
        #assuming NumLevels and I1,I2 dimensions are ok (i.e. we can divide I1,I2 to Numlevels pyramids)
        I1_pyr1_Down = cv2.pyrDown(I1_pyramids[i-1])
        I1_pyramids.append(I1_pyr1_Down)

        I2_pyr2_Down = cv2.pyrDown(I2_pyramids[i-1])
        I2_pyramids.append(I2_pyr2_Down)

    u = np.zeros(I2_pyramids[NumLevels-1].shape)
    v = np.zeros(I2_pyramids[NumLevels-1].shape)
    for i in range(NumLevels-1,0,-1):
        I2_warp = WarpImage(I2_pyramids[i],u,v)
        for j in range(Maxlter):
            (du,dv) = LucasKanadeStep(I1_pyramids[i],I2_warp,WindowSize)
            u = u + du
            v = v + dv
            I2_warp = WarpImage(I2_pyramids[i],u,v)
        if (i!=0):
            u = upsample2_array(u, I2_pyramids[i-1].shape[1], I2_pyramids[i-1].shape[0])
            v = upsample2_array(v, I2_pyramids[i-1].shape[1], I2_pyramids[i-1].shape[0])

    return u,v


def upsample2_array(arr,h,w):
    upsampled_arr=cv2.pyrUp(arr,dstsize=(h,w))
    return upsampled_arr*2


def LucasKanadeVideoStabilization(InputVid,WindowSize,MaxIter,NumLevels):
    cap = cv2.VideoCapture(InputVid)
    fourcc, fps, out_size = extract_video_params(cap)
    out = cv2.VideoWriter('StabilizedVid_204356315_200846038.avi', fourcc, fps, out_size, isColor=False)
    frameNum=0

    while (cap.isOpened()):
        ret, frame = cap.read()
        frameNum+=1
        if ret == True:
            print("Frame {0} is processed..".format(frameNum))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if frameNum==1:
                previousFrame = frame
                u_history = np.zeros(frame.shape)
                v_history = np.zeros(frame.shape)
                ones_matrix = np.ones(frame.shape)
                out.write(frame)
            else:
                (u, v) = LucasKanadeOpticalFlow(previousFrame,frame, WindowSize, MaxIter, NumLevels)
                u_history = u_history + u
                v_history = v_history + v
                # warp from k+1 to 1
                kk1_warp = WarpImage(frame, int(u_history.mean()) * ones_matrix, int(v_history.mean()) * ones_matrix)
                new=kk1_warp
                previousFrame = frame
                new = np.uint8(new)
                out.write(new)
        else:
            break
    free_video(cap, out)
    return


def  extract_video_params(cap):
  # Define the codec and create VideoWriter object
  fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Define video codec
  fps = int(cap.get(cv2.CAP_PROP_FPS))
  width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  #####
  height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  #####
  out_size = (width, height)

  return fourcc, fps, out_size


def free_video(cap,out):
  # Release everything if job is finished
  cap.release()
  out.release()
  cv2.destroyAllWindows()
  return