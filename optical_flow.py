import numpy as np
import cv2 as cv
import argparse
import math as mt

#camera intrinsic parameters
f = 3.67 #(mm)
sx = 3.98 * 10**(-3) #(mm)
sy = 3.98 * 10**(-3)
cx = 320
cy = 240

h_cam = 60 #(cm)

def get_thetas(u,v):
    
    x = (v-cx)*sx
    y = (u-cy)*sy
    theta_x = mt.atan(x/f)
    theta_y = mt.atan(y/f)
    if(theta_x*180/mt.pi) >= 90:
        print('angulo invalido')
    return theta_x, theta_y, x, y

def get_3D(theta_x):
    alpha = mt.pi - theta_x
    Z = mt.tan(alpha) * h_cam
    # X = Z*x/f
    # Y = Z*y/f
    return Z
# parser = argparse.ArgumentParser(description='This sample demonstrates Lucas-Kanade Optical Flow calculation. \
#                                               The example file can be downloaded from: \
#                                               https://www.bogotobogo.com/python/OpenCV_Python/images/mean_shift_tracking/slow_traffic_small.mp4')
# parser.add_argument('image', type=str, help='path to image file')
# args = parser.parse_args()
cap = cv.VideoCapture(0)
# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )
# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
# Create some random colors
color = np.random.randint(0,255,(100,3))
# Take first frame and find corners in it

while(1):
    ret, old_frame = cap.read()
    old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
    p0 = cv.goodFeaturesToTrack(old_gray, mask = None, **feature_params).astype('float32',casting='same_kind')
    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)
    ret,frame = cap.read()
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # calculate optical flow
    p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    # Select good points
    good_new = p1[st==1]
    good_old = p0[st==1]

    if good_new != good_old:
        theta_x, theta_y = get_thetas(p0)
        #depois publicamos esses valores nos servos, então calculamos as coordenadas 3D
    
    #depois que a bola está alinhada ao centro da câmera, calculamos as coordenadas 3D da bola

    Z = get_3D(theta_x)

    # draw the tracks
    for i,(new,old) in enumerate(zip(good_new, good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        mask = cv.line(mask, (a,b),(c,d), color[i].tolist(), 2)
        frame = cv.circle(frame,(a,b),5,color[i].tolist(),-1)
        frame = cv.circle(frame,(cy,cx), 15, (0,0,255), -1)
    img = cv.add(frame,mask)
    cv.imshow('frame',img)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break
    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1,1,2)



