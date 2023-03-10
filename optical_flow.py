import numpy as np
import cv2 as cv
import argparse
import math as mt
import rospy 
from std_msgs.msg import Float32MultiArray
from yolofinal.detect import main, run, parse_opt
import serial
import time
# from yolofinal.utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
#                            increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
ser = serial.Serial(
    port='/dev/ttyUSB0',
    baudrate=9600,
    timeout=1.5)
#camera intrinsic parameters
f = 3.67 #(mm)
sx = 3.98 * 10**(-3) #(mm)
sy = 3.98 * 10**(-3)
cx = 320
cy = 240

h_cam = 68 #(cm)
euler = np.identity(3)
def get_thetas(u,v):
    
    x = (v-cx)*sx
    y = (u-cy)*sy
    theta_x = int(np.arctan2(x,f)*180/mt.pi)
    print(x,theta_x,np.arctan2(x,f))
    theta_y = int(np.arctan2(y,f)*180/mt.pi)
    if(theta_y) >= 90:
        print('angulo invalido')
    return theta_x, theta_y

def get_3D(pitch):
    # print(theta_y)
    alpha = mt.pi/2 - pitch
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
Z = 0
pitch = 0
theta_y = 0
ser.write(bytes('30y', 'utf-8'))
while(1):
    ret, old_frame = cap.read()
    # print(np.shape(old_frame))
    old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
    cv.imwrite('old_frame.jpg',old_frame)
    # p0 = cv.goodFeaturesToTrack(old_gray, mask = None, **feature_params).astype('float32',casting='same_kind')
    # print(np.shape(p0))
    opt = parse_opt()
    xx,yy = main(opt)
    p0 = np.array([int(xx), int(yy)]).reshape(1,1,2).astype('float32')
    print(p0)
    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)
    ret,frame = cap.read()
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # calculate optical flow
    p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    # Select good points
    good_new = p1[st==1]
    good_old = p0[st==1]
    # print(good_new,good_old)
    # if (good_new.any()) and (good_old.any()):
    if (xx != cx) or (yy!= cy):
        theta_x, theta_y = get_thetas(int(yy),int(xx))
        ser.write(bytes(str(theta_y)+'y', 'utf-8'))
        time.sleep(0.15)
        ser.write(bytes(str(-theta_x)+'x', 'utf-8'))
        print(theta_x,theta_x,'theeeeeeta')

        # break
        
        
        
        #print(theta_x,theta_y)
        #break
        # print("ok")

        print(theta_y,theta_x,"theeeeeeeeeeeeeeta")
        cxx = mt.cos(-theta_x*np.pi/180)
        sxx = mt.sin(-theta_x*np.pi/180)
        cyy = mt.cos(theta_y*np.pi/180)
        syy = mt.sin(theta_y*np.pi/180)
        euler = np.array([[cxx*cyy, -sxx,cxx*syy],[sxx*cyy,cxx,sxx*syy]
                                ,[-syy,0,cyy]])@euler
        pitch = np.arctan2(-euler[2,0],mt.sqrt(euler[2,1]**2+euler[2,2]**2))
        # pitch = pitch + theta_y*np.pi/180
    # if p1 == :
    if theta_y == 0:
        Z = get_3D(pitch)


        print(Z,pitch,'zzzzzzzzzz')

    #     #break
    #     #depois publicamos esses valores nos servos, ent??o calculamos as coordenadas 3D
    
    # #depois que a bola est?? alinhada ao centro da c??mera, calculamos as coordenadas 3D da bola

    

    # draw the tracks
    for i,(new,old) in enumerate(zip(good_new, good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        mask = cv.line(mask, (a,b),(c,d), color[i].tolist(), 2)
        frame = cv.circle(frame,(a,b),5,color[i].tolist(),-1)
        frame = cv.circle(frame,(int(cx),int(cy)), 15, (0,0,255), -1)
        cv.putText(img=frame,text=str(Z), org=(int(xx), int(yy)), fontFace=cv.FONT_HERSHEY_SCRIPT_COMPLEX, fontScale=1, color=(255,255,0), 
        thickness=3)
    img = cv.add(frame,mask)
    cv.imshow('frame',img)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break
    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1,1,2)



