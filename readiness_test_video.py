import time
import cv2
import numpy as np
import imutils


# wait for the camera to warm up
cap = cv2.VideoCapture('Two disks.mp4')
# cap.set(3, 720)
# cap.set(4, 1280)


fgbg = cv2.createBackgroundSubtractorMOG2(history=50,detectShadows=False)
fgbg.setBackgroundRatio(0.9)

# loop over the frames from the video stream
while True:

	rect, frame = cap.read()



	fgmask = fgbg.apply(frame)

	cv2.imshow("Raw Mask",imutils.resize(fgmask, width=500))

	kernel = np.ones((14,14),np.uint8)  
	fgmask = cv2.erode(fgmask,kernel,iterations = 1)
	cv2.imshow("Errosion",imutils.resize(fgmask, width=500))
	kernel = np.ones((40,40),np.uint8)
	fgmask = cv2.dilate(fgmask,kernel,iterations = 1)
	# fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)

	cv2.imshow("Mask",imutils.resize(fgmask, width=500))
	
	# Z = np.float32(fgmask)
	# criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
	# ret,label,center=cv2.kmeans(Z,2,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
	# center = np.uint8(center)
	# res = center[label.flatten()]
	# res2 = res.reshape((fgmask.shape))
	# cv2.imshow('res2',imutils.resize(res2, width=500))

	contours, hierarchy= cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	contour_list = []
	hull_list = []
	for contour in contours:
		approx = cv2.approxPolyDP(contour,0.01*cv2.arcLength(contour,True),True)
		area = cv2.contourArea(contour)
		if (area > 1000):
			contour_list.append(contour)
			hull = cv2.convexHull(contour)
			hull_list.append(hull)
			
	# cnts = sorted(contour_list, key=cv2.contourArea, reverse=True)[:3]
	
	n = len(hull_list)
	cv2.putText(frame, str(n), (100,100), cv2.FONT_HERSHEY_SIMPLEX, 4, (0,255,0), 5)

	try:
		for i in hull_list:
			(x, y, w, h) = cv2.boundingRect(i) 
			cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
	except IndexError:
		print('[INFO] No Motion Detected')

	# cv2.drawContours(frame, cnts, -1, (255,0,0), 3)
	cv2.drawContours(frame, hull_list, -1, (0,0,255), 3)

	# show the output frame
	cv2.imshow("Frame", imutils.resize(frame, width=500))

	key = cv2.waitKey(25) & 0xFF
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

cap.release()
cv2.destroyAllWindows()