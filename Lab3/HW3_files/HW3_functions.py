import numpy as np

def predictParticles(S_next_tag):
    #INPUT  = S_next_tag (previously sampled particles)
    #OUTPUT = S_next (predicted particles. weights and CDF not updated yet)
    N=100
    m,n=S_next_tag.shape
    S_next=np.zeros((m,n))
    noise_sigma=4
    noise=np.random.normal(0,noise_sigma,(m,N))
    #width and height cannot change
    noise[2:4,:]=0
    for j in range(0,n): #column scan
        S_next[:,j]=S_next_tag[:,j]
        S_next[0,j]+=S_next_tag[4,j]
        S_next[1,j]+=S_next_tag[5,j]
        S_next[:,j]+=noise[:,j]
        #Xc,Yc are coordinates, must be integers, maybe Xv,Yv, too. easy to modify later if we wish..
        S_next[0:2,j]=S_next[0:2,j].round()
    return S_next

def compNormHist(I, S):
    #INPUT  = I (image) AND s (1x6 STATE VECTOR, CAN ALSO BE ONE COLUMN FROM S)
    #OUTPUT = normHist (NORMALIZED HISTOGRAM 16x16x16 SPREAD OUT AS A 4096x1 VECTOR.
    #NORMALIZED = SUM OF TOTAL ELEMENTS IN THE HISTOGRAM = 1)
    #assuming: I is an RGB image
    Xc=int(S[0])
    Yc=int(S[1])
    width_2=int(S[2])
    height_2=int(S[3])
    histogram=[]
    I_subportion = np.zeros((2*width_2,2*height_2, 3))
    for channel in range(0,I.shape[2]):
        I_subportion[:,:,channel]=I[Xc-width_2:Xc+width_2,Yc-height_2:Yc+height_2,channel]
        I_subportion[:, :, channel]=quantizise_4(I_subportion[:, :, channel],16)
        #bug: why hist has no pixels with 15. I_subportion[:, :, channel] has 15 element
        hist,bins=(np.histogram(I_subportion[:, :, channel], bins=range(0,16), density=False))
        #fix edges from right side
        hist = np.pad(hist, (0, 16 - len(hist)), mode='constant')
        histogram.append(hist)
    normHist=flatten_histogram(np.asarray(histogram))
    normHist=np.true_divide(normHist,float(np.sum(normHist)))
    return normHist

def quantizise_4(img, N):
    # uniform quantization of an image to values [0,N]
    min_val = np.min(img)
    max_val = np.max(img)
    quant_range = (max_val - min_val) / N
    quant_uniform = np.floor((np.floor_divide(img - min_val, quant_range)))
    # explanation: we normalize the values of the image to the window quant_uniform, and round down
    return quant_uniform

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
    return


def showParticles(I, S):
    #INPUT = I (current frame), S (current state vector)
    #W (current weight vector), i (number of current frame)
    return