import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2

def predictParticles(S_next_tag): 
    #INPUT  = S_next_tag (previously sampled particles)
    #OUTPUT = S_next (predicted particles. weights and CDF not updated yet)
    N=100
    m,n=S_next_tag.shape
    S_next=np.zeros((m,n))
    noise_sigma=0.2 #was 2#############################################
    noise=np.random.normal(0,noise_sigma,(m,N))
    #width and height cannot change
    noise[2:4,:]=0
    for j in range(0,n): #column scan
        S_next[:,j]=S_next_tag[:,j]
        S_next[0,j]+=S_next_tag[4,j]#Xvelocity
        S_next[1,j]+=S_next_tag[5,j]#Yvelocity
        S_next[:,j]+=noise[:,j]
        #Xc,Yc are coordinates, must be integers, maybe Xv,Yv, too. easy to modify later if we wish..
        S_next[0:2,j]=S_next[0:2,j].round()
    return S_next

def compNormHist(I, S):
    #INPUT  = I (image) AND s (1x6 STATE VECTOR, CAN ALSO BE ONE COLUMN FROM S)
    #OUTPUT = normHist (NORMALIZED HISTOGRAM 16x16x16 SPREAD OUT AS A 4096x1 VECTOR.
    #NORMALIZED = SUM OF TOTAL ELEMENTS IN THE HISTOGRAM = 1)
    #assuming: I is an RGB image
    m,n,channels=I.shape
    Q_factor=16
    Xc, Yc, width_2, height_2=extract_params(S)
    histogram=[]
    #not enough - needs to take care of the case the window is outside image boundaries
    #I_subportion = np.zeros((2*width_2,2*height_2, channels),dtype=int) BAD XY ORIENTATION
    I_subportion = np.zeros((2*height_2,2*width_2, channels),dtype=int)
    for channel in range(0,channels):
        #check my m and n values. this is the only part that was not tested
        I_subportion[:, :, channel] = I[Yc - height_2:Yc + height_2, Xc - width_2:Xc + width_2 , channel]
        I_subportion[:, :, channel]=quantizise_4(I_subportion[:, :, channel],Q_factor-1)
        #bug: why hist has no pixels with 15. I_subportion[:, :, channel] has 15 element
        hist,bins=np.histogram(I_subportion[:, :, channel],range=(0,Q_factor-1), bins=Q_factor, density=False)
        #fix edges from right side - necessary only for the case there are not 15 values. but i think not really possible...
        hist = np.pad(hist, (0, Q_factor - len(hist)), mode='constant')
        histogram.append(hist)
    normHist=flatten_histogram(np.asarray(histogram))
    normHist=np.true_divide(normHist,float(np.sum(normHist)))#normalize by sum
    return normHist


def extract_params(S):
    Xc = int(S[0])
    Yc = int(S[1])
    width_2 = int(S[2])
    height_2 = int(S[3])
    return Xc,Yc,width_2,height_2


def quantizise_4(img, N):
    # uniform quantization of an image to values [0,N]
    #min_val = np.min(img)#ORIGINAL
    #max_val = np.max(img)#ORIGINAL
    #quant_range = (max_val - min_val) / N
    #quant_uniform = np.floor((np.floor_divide(img - min_val, quant_range))).astype(int)
    # explanation: we normalize the values of the image to the window quant_uniform, and round down
    quant = int(255/N)
    for j in range(0,N):
        mask = (img <= (j+1)*quant) & (img >= (j)*quant) #element wise AND
        img[mask] = j
    return(img)

def flatten_histogram(histogram):
    #assumes a list of 3 histograms we want to flatten, assume histogram.shape=(n,channels)
    channels,n=histogram.shape
    flat_n=np.power(n,channels)
    flat_hist=np.repeat(histogram[0],256)
    i=0
    while (i<flat_n):
        for j in range(0,n):
            if (histogram[1, j]==0):  flat_hist[i]=0
            else: flat_hist[i] *= histogram[1, j]
            for h in range (0,n):
                if (histogram[2,h] == 0):  flat_hist[i] = 0
                else:flat_hist[i]*=histogram[2,h]
                i+=1
    return flat_hist


def compBatDist(p, q):
    #INPUT  = p , q (2 NORMALIZED HISTOGRAM VECTORS SIZED 4096x1)
    # OUTPUT = THE BHATTACHARYYA DISTANCE BETWEEN p AND q (1x1)
    distance=0
    for i in range(1,4096):
        distance+=np.sqrt(p[i]*q[i])
    distance*=20.0
    distance=np.exp(distance)
    return distance

def compute_CDF(W):
    #checked! works!
    return np.cumsum(W)

def sampleParticles(S_prev, C):
    #INPUT  = S_prev (PREVIOUS STATE VECTOR MATRIX), C (CDF)
    #OUTPUT = S_next_tag (NEW X STATE VECTOR MATRIX)
    S_next_tag=np.zeros(S_prev.shape)
    N=100
    for n in range(0,N):
        r=np.random.uniform()
        j = np.where(C == np.min(C[C >= r]))[0][0]
        S_next_tag[:, n] = S_prev[:, j]
    return S_next_tag


def showParticles(I, S, W, frame_number, ID):
    #INPUT = I (current frame), S (current state vector)
    #W (current weight vector), i (number of current frame)
    I_RGB=cv2.cvtColor(I,cv2.COLOR_BGR2RGB)
    maximal_filter_index=np.argmax(W)
    average_filter_index=find_nearest(W,np.average(W))
    S_maximal_filter=S[:,maximal_filter_index]
    S_average_filter=S[:,average_filter_index]
    Xc, Yc, width_2, height_2=extract_params(S_maximal_filter)
    maximal_rect = patches.Rectangle((Xc - width_2, Yc - height_2), width_2 * 2, height_2 * 2, linewidth=1,
                                     edgecolor='r', facecolor='none')
    Xc, Yc, width_2, height_2=extract_params(S_average_filter)
    average_rect = patches.Rectangle((Xc - width_2,Yc - height_2), width_2 * 2, height_2 * 2, linewidth=1,
                                      edgecolor='g', facecolor='none')

    fig,ax=plt.subplots(1)
    plt.title('{0}- Frame number = {1}'.format(ID,frame_number))
    ax.imshow(I_RGB)
    #plt.ion()
    ax.add_patch(maximal_rect)
    ax.add_patch(average_rect)
    plt.show()
    #plt.show(block = False)



    return

'''
def arg_median(a):
    if len(a) % 2 == 1:
        return np.where(a == np.median(a))[0][0]
    else:
        l,r = len(a) // 2 - 1, len(a) // 2
        left = np.partition(a, l)[l]
        right = np.partition(a, r)[r]
        return [np.where(a == left)[0][0], np.where(a == right)[0][0]]
    '''

def find_nearest(a, a0):
    "Element in nd array `a` closest to the scalar value `a0`"
    idx = np.abs(a - a0).argmin()
    return np.where(a==a.flat[idx])[0][0]