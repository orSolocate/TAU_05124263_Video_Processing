import numpy as np
import cv2

###############################################Part_2#######################################################

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

def Q2A(url):
  cap = cv2.VideoCapture(url)
  fourcc, fps, out_size= extract_video_params(cap)
  out = cv2.VideoWriter('Vid1_Binary.avi', fourcc, fps, out_size, isColor=False)

  while (cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:  # If we succeeded in reading a frame from the input video object
      frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      ret, frame = cv2.threshold(frame,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
      out.write(frame)

    else: break

  free_video(cap,out)
  return


def Q2B(url):
  cap = cv2.VideoCapture(url)
  fourcc, fps, out_size= extract_video_params(cap)
  out = cv2.VideoWriter('Vid2_Grayscale.avi', fourcc, fps, out_size, isColor=False)

  while (cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
      grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      out.write(grayFrame)
    else: break

  free_video(cap,out)
  return


def Q2C(url):
  cap = cv2.VideoCapture(url)
  fourcc, fps, out_size= extract_video_params(cap)
  out = cv2.VideoWriter('Vid3_Sobel.avi', fourcc, fps, out_size, isColor=False)

  while (cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:  # If we succeeded in reading a frame from the input video object
      frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      sobelx = cv2.Sobel(frame, cv2.CV_8U, 1, 0, ksize=5)  # x
      sobely = cv2.Sobel(frame, cv2.CV_64F, 0, 1, ksize=5)  # y
      abs_grad_x = cv2.convertScaleAbs(sobelx)
      abs_grad_y = cv2.convertScaleAbs(sobely)
      grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0) #can be also used according to https://docs.opencv.org/3.4/d2/d2c/tutorial_sobel_derivatives.html
      out.write(grad)
    else: break

  free_video(cap,out)
  return