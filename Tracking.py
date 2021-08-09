#! Python 3
# Measure the progress at a construction site using a camera, by tracking 
# the floor area that the concrete has covered.

import  cv2
import os
import numpy as np
import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

def tracking(vid_dir: str) -> None:

    cap = cv2.VideoCapture(vid_dir)

    # Frame number
    frame_no = 1

    logging.info('"Left click" to select a point.')
    logging.info('"Right click" to cancel the last selected point.')
    logging.info('"Middle click" to join the last selected point and first point to close the loop.')
    logging.info('"S" to save the selected ROI and mask of ROI.')
    logging.info('"Esc" to get out of ROI selection.')

    while cap.isOpened():
        ret, frame = cap.read()

        pts=[]

        # At the first frame  we will define our Region of Interest
        if frame_no == 1:

            def draw_roi(event, x, y, flags, param):            
                img = frame.copy()

                if event == cv2.EVENT_LBUTTONDOWN: # Left click, select point
                    pts.append((x, y))  

                if event == cv2.EVENT_RBUTTONDOWN: # Right click to cancel the last selected point
                    pts.pop()  

                if event == cv2.EVENT_MBUTTONDOWN: # Connect the last and first points to create a close loop (ROI)
                    mask = np.zeros(frame.shape, np.uint8) 
                    points = np.array(pts, np.int32)
                    points = points.reshape((-1, 1, 2))

                    mask = cv2.polylines(mask, [points], True, (255, 255, 255), 2)
                    mask2 = cv2.fillPoly(mask.copy(), [points], (255, 255, 255)) # for ROI
                    mask3 = cv2.fillPoly(mask.copy(), [points], (0, 255, 0)) # for displaying images on the desktop

                    ROI = cv2.bitwise_and(mask2, frame) # Get the ROI  
                    cv2.imshow("ROI", ROI) # To check if selected ROI is good or not

                    key = cv2.waitKey(0) & 0xFF

                    # If the selected ROI is good then save the ROI and mask
                    if key == ord('s'):
                        cv2.imwrite('ROI.jpg', ROI)
                        cv2.imwrite('mask.jpg', mask2)
                        print('Mask and ROI saved.')

                if len(pts) > 0:
                    # Draw the point in image when we click left mouse button
                    cv2.circle(img, pts[-1], 3, (0, 0, 255), -1)

                if len(pts) > 1:
                    # Draw line connecting last point and new point
                    for i in range(len(pts) - 1):
                        cv2.circle(img, pts[i], 5, (0, 0, 255), -1) # x ,y is the coordinates of the mouse click place
                        cv2.line(img=img, pt1=pts[i], pt2=pts[i + 1], color=(255, 0, 0), thickness=2)

                cv2.imshow('image', img)
            cv2.imshow('image', frame)
            cv2.setMouseCallback('image', draw_roi)

            frame_no += 1 # To get out of ROI function and continue the video
    
            # Get out of the ROI selectoin after saving ROI by pressing Esc
            while True:
                key = cv2.waitKey(0) & 0xFF
                if key == 27:
                    break
            cv2.destroyAllWindows()
            
        if not ret:
            print("Can't receive frame. Exiting.")
            break


        ROI = cv2.imread('ROI.jpg')
        mask = cv2.imread('mask.jpg')
        
        # Get ROI for next frames
        frame_ROI = cv2.bitwise_and(mask, frame)
        frame_hsv = cv2.cvtColor(frame_ROI, cv2.COLOR_BGR2HSV)

        # This will track the concret in the video.To find the HSV value, I used trackbar.py.
        # Code of the trackbar is included with this folder.
        thresh = cv2.inRange(frame_hsv, (24, 1, 1), (102, 255, 255))

        kernel = np.ones((13,13), np.uint8)
        thresh = cv2.erode(thresh, kernel)
        kernel = np.ones((20,20), np.uint8)
        thresh = cv2.dilate(thresh, kernel)
        #cv2.imshow('Changes', thresh)

        # To show the change in the original image as a green overlay, change the mask to green color
        mask_RGB = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)
        mask_RGB[np.where((mask_RGB==[255,255,255]).all(axis=2))] = [0,255,0]

        # Convert original mask to grayscale so that we can compare it with the threshold we get from 
        # inrange function. By doing that we can get the progress.
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        progress = np.sum(thresh)/np.sum(mask) * 100
        print(f'Progress: {progress: .2f}%')
        frame = cv2.putText(frame, f'Progress: {progress:.2f}%', (350,630), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 4, cv2.LINE_AA)

        mask_in_original = cv2.addWeighted(src1=frame, alpha=1, src2=mask_RGB, beta=0.4, gamma=0)
        cv2.imshow('Changes as mask', mask_in_original)

        if cv2.waitKey(1000) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    vid_dir = './data/concreting.mp4'
    
    tracking(vid_dir)