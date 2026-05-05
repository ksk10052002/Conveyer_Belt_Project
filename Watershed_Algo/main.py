import cv2
import numpy as np

cap = cv2.VideoCapture(0)

counts = {
    "Triangle": 0,
    "Square": 0,
    "Rectangle": 0,
    "Circle": 0
}

line_y = 150  # inside ROI

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ---------------- ROI (IMPORTANT) ----------------
    roi = frame[200:400, 100:600]   # adjust based on your camera

    # gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # blur = cv2.GaussianBlur(gray, (5,5), 0)

    # # Better threshold than Canny for object detection
    # _, thresh = cv2.threshold(blur, 120, 255, cv2.THRESH_BINARY_INV)

    # contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
    #                                cv2.CHAIN_APPROX_SIMPLE)



    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # 🔥 Improve contrast (important for white objects)
    gray = cv2.equalizeHist(gray)

    blur = cv2.GaussianBlur(gray, (5,5), 0)

    # 🔥 Adaptive threshold (handles white objects)
    thresh = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11, 2
    )

    # 🔥 Edge detection (for phone, strong edges)
    edges = cv2.Canny(blur, 50, 150)

    # 🔥 Combine both
    combined = cv2.bitwise_or(thresh, edges)

    # 🔥 Fix broken edges
    kernel = np.ones((3,3), np.uint8)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 🔥 Find contours on combined image
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL,
                                  cv2.CHAIN_APPROX_SIMPLE)


    for cnt in contours:
        area = cv2.contourArea(cnt)

        # -------- AREA FILTER --------
        if 800 < area < 8000:

            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

            x, y, w, h = cv2.boundingRect(approx)

            cx = int(x + w/2)
            cy = int(y + h/2)

            vertices = len(approx)

            # -------- SHAPE DETECTION --------
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

            else:
                continue  # ignore weird shapes

            # -------- LINE COUNT --------
            if abs(cy - line_y) < 10:
                counts[shape] += 1

            # -------- DRAW --------
            cv2.putText(roi, shape, (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

            cv2.drawContours(roi, [approx], -1, (0,255,0), 2)
            cv2.circle(roi, (cx, cy), 4, (0,0,255), -1)

    # -------- DRAW LINE --------
    cv2.line(roi, (0, line_y), (500, line_y), (255,0,0), 2)

    # -------- DISPLAY COUNTS --------
    y_offset = 20
    for shape_name, count in counts.items():
        cv2.putText(frame, f"{shape_name}: {count}",
                    (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255,0,0), 2)
        y_offset += 25

    cv2.imshow("ROI", roi)
    cv2.imshow("Thresh", thresh)
    cv2.imshow("Edges", edges)
    cv2.imshow("Combined", combined)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()