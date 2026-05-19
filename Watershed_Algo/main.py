import cv2
import math
import numpy as np

cap = cv2.VideoCapture(0)


while True:

    ret, frame = cap.read()

    if not ret:
        break

    # ---------------- ROI ----------------

    roi = frame[120:420, 80:560]

    # -------------------------------------
    
    #____Converting to GreyScale_____________#

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    #_______Adding Gausian Blur_____#

    blur = cv2.GaussianBlur(gray, (5,5), 0)

    #_____Adding threshold_____#

    _, thresh = cv2.threshold(
        blur,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    #____________Removing Noise____________#
    
    kernel = np.ones((3,3), np.uint8)

    opening = cv2.morphologyEx(
        thresh,
        cv2.MORPH_OPEN,
        kernel,
        iterations=2
    )

    #------------------------------------------

    #_________Sure BAckground____________#

    sure_bg = cv2.dilate(
        opening, kernel,iterations=3
    )

    ###----------------------------------------

    #_________Distance Transform__________#

    dist_transform = cv2.distanceTransform(
        opening,
        cv2.DIST_L2,
        5
    )

    #----------------------------------------

    # #_________Distance_Transform_Normalize__________#

    # dist_display = cv2.normalize(
    # dist_transform,
    # None,
    # 0,
    # 255,
    # cv2.NORM_MINMAX
    # )

    # dist_display = np.uint8(dist_display)

    # #----------------------------------------


    #___________Sure Foreground___________#

    _, sure_fg = cv2.threshold(
        dist_transform,
        0.5 * dist_transform.max(),
        255,
        0
    )

    sure_fg = np.uint8(sure_fg)

    #==========================================

    #__________Unknown Region______________#

    unknown = cv2.subtract(
        sure_bg,
        sure_fg
    )

    #========================================

    #___________Marking Sure_FG___________#

    _, markers = cv2.connectedComponents(
        sure_fg
    )

    markers = markers + 1

    markers[unknown == 255] = 0

    #----------------------------------------

    #_______Applying Watershed Algorithm______#

    markers = cv2.watershed(
        roi,
        markers
    )

    #--------------------------------------------

    #__________Boundary Color________________#

    roi[markers == -1] = [0, 0, 255]

    #--------------------------------------------


    # ---------------- EXTRACT OBJECTS ----------------

    for label in np.unique(markers):

        # Ignore background and boundary

        if label == 1 or label == -1:
            continue

        # Create mask

        mask = np.zeros(gray.shape, dtype="uint8")

        mask[markers == label] = 255

        # Find contours of separated object

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        for cnt in contours:

            area = cv2.contourArea(cnt)

            if area > 500:

                # Polygon approximation

                peri = cv2.arcLength(cnt, True)

                approx = cv2.approxPolyDP(
                    cnt,
                    0.02 * peri,
                    True
                )

                vertices = len(approx)

                # Bounding box

                x, y, w, h = cv2.boundingRect(approx)

                # Center point

                cx = int(x + w/2)
                cy = int(y + h/2)

                # ---------------- SHAPE DETECTION ----------------

                shape = "Unknown"

                if vertices == 3:

                    shape = "Triangle"

                elif vertices == 4:

                    ratio = w / float(h)

                    if 0.95 <= ratio <= 1.05:

                        shape = "Square"

                    else:

                        shape = "Rectangle"

                elif vertices > 6:

                    shape = "Circle"

                # -------------------------------------------------

                # Draw contour

                cv2.drawContours(
                    roi,
                    [approx],
                    -1,
                    (0,255,0),
                    2
                )

                # Draw rectangle

                cv2.rectangle(
                    roi,
                    (x,y),
                    (x+w,y+h),
                    (255,0,0),
                    2
                )

                # Draw center

                cv2.circle(
                    roi,
                    (cx, cy),
                    5,
                    (0,0,255),
                    -1
                )

                # Display shape

                cv2.putText(
                    roi,
                    shape,
                    (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255,255,0),
                    2
                )
    
    ## ---------------- finding the final objects CONTOURS ----------------

    # contours, _ = cv2.findContours(
    #     sure_fg,
    #     cv2.RETR_EXTERNAL,
    #     cv2.CHAIN_APPROX_SIMPLE
    # )

    # # ------------------------------------------


    # for cnt in contours:

    #     area = cv2.contourArea(cnt)

    #     if area > 500:

    #         x,y,w,h = cv2.boundingRect(cnt)

    #         cv2.rectangle(
    #             roi,
    #             (x,y),
    #             (x+w,y+h),
    #             (255,0,0),
    #             2
    #         )

    #-----------------------------------------------

    #____Showing Window________#

    cv2.imshow("ROI", roi)
    cv2.imshow("Threshold", thresh)
    cv2.imshow("Opening", opening)
    cv2.imshow("Distance Transform", dist_transform / dist_transform.max())
    cv2.imshow("Sure FG", sure_fg)
    cv2.imshow("Unknown", unknown)
    # cv2.imshow("Distance", dist_display)

    
    cv2.imshow("Full Frame", frame)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
            

           