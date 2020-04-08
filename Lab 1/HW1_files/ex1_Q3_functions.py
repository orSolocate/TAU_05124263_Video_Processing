import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

def divide_grid(arr, nrows, ncols):
    # based on: https://stackoverflow.com/questions/16856788/slice-2d-array-into-smaller-2d-arrays

    h, w = arr.shape
    return (arr.reshape(h//nrows, nrows, -1, ncols)
               .swapaxes(1,2)
               .reshape(-1, nrows, ncols))

def join_grid(grid,nrows,ncols):
    grows,gcols=grid.shape[1], grid.shape[2]
    horiz_blocks=int(ncols/grows)
    vertical_blocks=int(nrows/gcols)
    for j in range(0,vertical_blocks):
        for i in range(0,horiz_blocks):
            if i==0:
                temp = grid[(j*horiz_blocks+0)]
            else:
                temp=np.concatenate((temp,grid[j*horiz_blocks+i]),axis=1)
        if j==0:
            array=temp
        else: array=np.concatenate((array,temp),axis=0)
    return array

def zeroAllButMaxInArray(array):
    binary_array=np.zeros_like(array)
    max_idx=array.argmax()
    binary_array.flat[max_idx]=array.flat[max_idx]
    return binary_array


def myHarrisCornerDetector(IN,K,Threshold,use_grid=True):
    #assume IN is an RGB image
    IN_grey=cv2.cvtColor(IN, cv2.COLOR_RGB2GRAY)
    Ix,Iy=np.gradient(IN_grey)
    Ix2=np.multiply(Ix,Ix)
    Iy2=np.multiply(Iy,Iy)
    Sxx=cv2.GaussianBlur(Ix2,(5,5),0)
    Syy=cv2.GaussianBlur(Iy2,(5,5),0)
    Sxy=cv2.GaussianBlur(np.multiply(Ix,Iy),(5,5),0)
    R=Sxx*Syy- np.power(Sxy,2)-K*np.power(Sxx+Syy,2)
    R[np.abs(R)<Threshold]=0
    rows, cols = R.shape[0], R.shape[1]
    if (use_grid==True):    #assumes a 25x25 grid
        my_grid=divide_grid(R,25,25)
        for window in my_grid:
            window=zeroAllButMaxInArray(window)
        R=join_grid(my_grid,rows,cols)
    R[R != 0] = 1
    return R

def createCornerPlots(I1,I1_CORNERS,I2,I2_CORNERS):
    I1_dis=cv2.cvtColor(I1, cv2.COLOR_BGR2RGB)
    I2_dis = cv2.cvtColor(I2, cv2.COLOR_BGR2RGB)
    fig,ax=plt.subplots(1,2)
    for i in range(2):
        if i==0:
            ax[i].imshow(I1_dis)
            circles_locations = np.argwhere(I1_CORNERS == 1)
        elif i==1:
                ax[i].imshow(I2_dis)
                circles_locations = np.argwhere(I2_CORNERS == 1)
        for y,x in circles_locations:
            ax[i].add_patch(Circle((x,y),radius=1,color='red'))
    plt.show(block=True)
    return