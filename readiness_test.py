import time
import cv2
import numpy as np
import imutils

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

	

	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	# blur = cv2.GaussianBlur(gray, (5, 5), 0)
	blur = cv2.bilateralFilter(gray, 5, 175, 175)
	edge_detected_image = cv2.Canny(blur, 75, 200)
	
	cv2.imshow('Edge', imutils.resize(edge_detected_image, width=500))

	contours, hierarchy= cv2.findContours(edge_detected_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	contour_list = []
	for contour in contours:
		approx = cv2.approxPolyDP(contour,0.01*cv2.arcLength(contour,True),True)
		area = cv2.contourArea(contour)
		if ((len(approx) > 10) & (area > 30) ):
			contour_list.append(contour)

	# cnts = sorted(contour_list, key=cv2.contourArea, reverse=True)[:5]
	display = frame.copy()
	cv2.drawContours(display, contour_list, -1, (0, 255, 0), 2)


	fgmask = fgbg.apply(frame)
	fg = cv2.bitwise_and(cv2.cvtColor(fgmask, cv2.COLOR_GRAY2RGB),frame)
	fg_blur = cv2.bilateralFilter(fg, 5, 175, 175)

	# show the output frame
	cv2.imshow("Frame", imutils.resize(display, width=500))
	cv2.imshow("Foreground", imutils.resize(fg_blur, width=500))

	key = cv2.waitKey(1) & 0xFF
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

cap.release()
cv2.destroyAllWindows()