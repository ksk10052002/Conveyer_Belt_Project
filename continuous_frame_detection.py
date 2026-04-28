import cv2
import time

cap = cv2.VideoCapture(0)                                   #Intializes the webcam

counts = {
    "Triangle": 0,
    "Square": 0,
    "Rectangle": 0,
    "Circle": 0,
    "Pentagon": 0,
    "Hexagon": 0
}

while True:                                                 #Starts a loop to process the video stream frame by frame.

    ret, frame = cap.read()                                 # Grabs the current image (frame) from the camera.

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)          # Converts the frame to grayscale for easier processing.

    blur = cv2.GaussianBlur(gray, (5,5), 0)                 # Blurs the image to reduce noise.

    edges = cv2.Canny(blur, 50, 150)                        # Detects edges in the image.

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)   #Scans the edges image to find connected curves.

    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area > 1000:



################################ FOR SIMPLE SHAPES ################################

            # approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)

            # x, y, w, h = cv2.boundingRect(approx)

            # if len(approx) == 3:
            #     shape = "Triangle"
            # elif len(approx) == 4:
            #     shape = "Square/Rectangle"q 
            # elif len(approx) > 6:
            #     shape = "Circle"
            # else:
            #     shape = "Unknown"

######################## FOR DIFFERENT SHAPES /////////////////////

            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

            vertices = len(approx)

            if vertices == 3:
                shape = "Triangle"

            elif vertices == 4:
                # Check square vs rectangle
                x, y, w, h = cv2.boundingRect(approx)
                cv2.putText(frame, shape, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

                aspectRatio = float(w) / h

                if 0.95 <= aspectRatio <= 1.05:
                    shape = "Square"
                else:
                    shape = "Rectangle"

            elif vertices == 5:
                shape = "Pentagon"

            elif vertices == 6:
                shape = "Hexagon"

            elif vertices > 6:
                shape = "Circle"

            else:
                shape = "Unknown"

            if shape in counts:
                counts[shape] += 1
                time.sleep(0.5)

            cv2.drawContours(frame, [approx], -1, (0,255,0), 2)

    offset = 30
    for shape, count in counts.items():
        text = f"{shape}: {count}"
        cv2.putText(frame, text, (10, offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
        offset += 25

    cv2.imshow("Shape Detection", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows() 