#! Python 3
# Thresholding an image using HSV values.

import cv2
import numpy as np

cv2.namedWindow('Trackbar')

cv2.createTrackbar('H Lower','Trackbar',0,179,nothing)
cv2.createTrackbar('H Higher','Trackbar',179,179,nothing)
cv2.createTrackbar('S Lower','Trackbar',0,255,nothing)
cv2.createTrackbar('S Higher','Trackbar',255,255,nothing)
cv2.createTrackbar('V Lower','Trackbar',0,255,nothing)
cv2.createTrackbar('V Higher','Trackbar',255,255,nothing)


while(1):
    img = cv2.imread('frame_4.jpg')

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    hL = cv2.getTrackbarPos('H Lower','Trackbar')
    hH = cv2.getTrackbarPos('H Higher','Trackbar')
    sL = cv2.getTrackbarPos('S Lower','Trackbar')
    sH = cv2.getTrackbarPos('S Higher','Trackbar')
    vL = cv2.getTrackbarPos('V Lower','Trackbar')
    vH = cv2.getTrackbarPos('V Higher','Trackbar')

    LowerRegion = np.array([hL,sL,vL],np.uint8)
    upperRegion = np.array([hH,sH,vH],np.uint8)

    redObject = cv2.inRange(hsv,LowerRegion,upperRegion)

    kernal = np.ones((1,1),"uint8")

    red = cv2.morphologyEx(redObject,cv2.MORPH_OPEN,kernal)
    red = cv2.dilate(red,kernal,iterations=1)

    result=cv2.bitwise_and(img, img, mask = red)


    cv2.imshow("Image",result)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break