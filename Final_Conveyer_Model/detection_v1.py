import cv2
import time

cap = cv2.VideoCapture(0)

#---Object Count ---------#

count = 0

#--Counting Line Position----#

line_y = 100

while True:

    ret, frame = cap.read()

    #--------------ROI------------------#

    roi = frame[200:400, 100:600]

    #-----------------------------------

    #----Convert ROI to GREY SCALE----------#

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    #----------------------------------------

    #--Blurring Image to reduce NOISE---#

    blur = cv2.GaussianBlur(gray, (5,5), 0)

    #-----------------------------------

    #---------Thresholding----------#

    _, thresh = cv2.threshold(
        blur,
        120,
        255,
        cv2.THRESH_BINARY_INV
    )

    #---------------------------------

    #-----------CONTOUR_DETECTION-------#

    contour, _= cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    ) 

    #------------------------------------

    for cnt in contour:

        #---Find the Contour Area---#

        area = cv2.contourArea(cnt)

        #---Filtering the Area------#

        if area > 1000:

            #-- Bounding Rectangle -----#

            x, y, w, h = cv2.boundingRect(cnt)

            #---center point----#

            cx = int(x + w/2)
            cy = int(y + h/2)

            #--DRAW Contour--#

            cv2.drawContours(roi, [cnt], -1, (0,255,0), 2)

            #---Draw Rectangle ----#

            cv2.rectangle(roi, (x,y), (x+w, y+h), (255,0,0), 2)

            #--Draw Center Point----#

            cv2.circle(roi, (cx, cy), 5, (0,0,255), -1)

            #-----Counting of objects----#

            if abs(cy - line_y) < 10:

                count += 1

                time.sleep(0.3)

            #-----------------------

    #--------Draw Counting Line----------#

    cv2.line(
        roi,
        (0, line_y),
        (500, line_y),
        (0,255,255),
        2
    )

    #------ Display Count ---------#

    cv2.putText(
        frame,
        f"Count: {count}",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,0,255),
        2
    )

    # #------------------DRAW CONTOURS------------------

    # cv2.drawContours(roi, contour, -1, (0,255,0), 2)

    #-------------------------------------------------

    #------DRAW ROI Rectangle --------#
    
    cv2.rectangle(frame, (100, 200), (600, 400), (255,0,0), 2)

    #----SHOW Windows--------#

    cv2.imshow("Full Frame", frame)
    cv2.imshow("ROI", roi)
    cv2.imshow("Threshold", thresh)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()