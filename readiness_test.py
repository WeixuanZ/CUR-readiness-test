import time
import cv2
import numpy as np
import imutils

# clustering

# wait for the camera to warm up
print("[INFO] starting video stream...")
cap = cv2.VideoCapture(0)
cap.set(3, 720)
cap.set(4, 1280)
time.sleep(2.0)

fgbg = cv2.createBackgroundSubtractorMOG2(history=100,detectShadows=False)
fgbg.setBackgroundRatio(0.8)

# loop over the frames from the video stream
while True:

	rect, frame = cap.read()

	fgmask = fgbg.apply(frame)
	kernel = np.ones((5,5),np.uint8)
	fgmask = cv2.erode(fgmask,kernel,iterations = 1)
	kernel = np.ones((20,20),np.uint8)
	fgmask = cv2.dilate(fgmask,kernel,iterations = 1)

	cv2.imshow("Mask",imutils.resize(fgmask, width=500))

	contours, hierarchy= cv2.findContours(fgmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	contour_list = []
	for contour in contours:
		approx = cv2.approxPolyDP(contour,0.01*cv2.arcLength(contour,True),True)
		area = cv2.contourArea(contour)
		if (area > 500):
			contour_list.append(contour)
			
	cnts = sorted(contour_list, key=cv2.contourArea, reverse=True)[:3]
	
	try: 
		for i in cnts:
			(x, y, w, h) = cv2.boundingRect(i) 
			cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
	except IndexError:
		print('No Motion Detected')

	# show the output frame
	cv2.imshow("Frame", imutils.resize(frame, width=500))

	key = cv2.waitKey(1) & 0xFF
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

cap.release()
cv2.destroyAllWindows()