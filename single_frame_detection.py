import cv2

cap = cv2.VideoCapture(0)

counts = {
    "Triangle": 0,
    "Square": 0,
    "Rectangle": 0,
    "Circle": 0,
    "Pentagon": 0,
    "Hexagon": 0
}

tracked_objects = []

line_y = 300   # counting line

while True:

    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area > 1000:

            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

            x, y, w, h = cv2.boundingRect(approx)
            cx = int(x + w/2)
            cy = int(y + h/2)

            vertices = len(approx)

            if vertices == 3:
                shape = "Triangle"

            elif vertices == 4:
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

            is_new = True
            for (px, py) in tracked_objects:
                if abs(cx - px) < 20 and abs(cy - py) < 20:
                    is_new = False
                    break

            if abs(cy - line_y) < 10 and is_new:
                if shape in counts:
                    counts[shape] += 1
                    tracked_objects.append((cx, cy))

            cv2.putText(frame, shape, (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

            cv2.drawContours(frame, [approx], -1, (0,255,0), 2)
            cv2.circle(frame, (cx, cy), 5, (0,0,255), -1)

    cv2.line(frame, (0, line_y), (640, line_y), (0,0,255), 2)

    offset = 30
    for shape_name, count in counts.items():
        text = f"{shape_name}: {count}"
        cv2.putText(frame, text, (10, offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
        offset += 25

    cv2.imshow("Shape Detection", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()