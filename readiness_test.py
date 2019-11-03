import time
import cv2
import numpy as np
import imutils

print("[INFO] starting video stream...")
cap = cv2.VideoCapture(0)
cap.set(3, 720)
cap.set(4, 1280)
time.sleep(2.0)

# fgbg = cv2.createBackgroundSubtractorMOG2(history=300,detectShadows=False)
# fgbg.setBackgroundRatio(0.7)


# loop over the frames from the video stream
while True:
    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 400 pixels
    rect, frame = cap.read()

    # fgmask = fgbg.apply(frame)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    ret, thresh = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY) # isolate the bright area

    paper = cv2.bitwise_and(thresh, thresh, mask=skinMask_inv) # remove the skin

    thresh2 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)
    paper = cv2.bitwise_and(paper,paper, mask=thresh2) # add black edges surrounding the paper

    cnts = cv2.findContours(paper.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]


    # foreGround = cv2.bitwise_and(paper, paper, mask=fgmask)
    # blur2 = cv2.GaussianBlur(foreGround, (5, 5), 0)
    # edged = cv2.Canny(gray, 50, 150)

    # cnts2 = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # cnts2 = imutils.grab_contours(cnts2)
    # cnts2 = sorted(cnts2, key=cv2.contourArea, reverse=True)[:5]

    c = cnts[0]
    # approximate the contour
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)

    # c2 = cnts2[0]
    # peri2 = cv2.arcLength(c2, True)
    # approx2 = cv2.approxPolyDP(c2, 0.02 * peri, True)
    # cv2.drawContours(foreGround, [approx2], -1, (0, 255, 0), 2)

    if len(approx) == 4:
        screenCnt = approx
        image = cv2.bitwise_and(frame,frame,mask=skinMask_inv)
        #image=frame
        cv2.drawContours(frame, [approx], -1, (0, 255, 0), 2)
        cv2.drawContours(image, [approx], -1, (0, 255, 0), 2)
    else:
        cv2.drawContours(frame, [approx], -1, (0, 0, 255), 2)

    try:
        image
    except NameError:
        image = None

    # show the output frame
    cv2.imshow("Frame", imutils.resize(frame, width=500))

    key = cv2.waitKey(1) & 0xFF
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()