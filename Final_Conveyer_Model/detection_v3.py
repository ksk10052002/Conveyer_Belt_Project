import cv2
import math

cap = cv2.VideoCapture(0)

# Camera resolution

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# ---------------- SHAPE COUNTS ----------------

counts = {
    "Triangle": 0,
    "Square": 0,
    "Rectangle": 0,
    "Circle": 0
}

# ---------------- TRACKER ----------------

tracked_objects = {}

object_id = 0

# ---------------- COUNTING LINE ----------------

line_y = 150

# ----------------------------------------------

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # ---------------- ROI ----------------

    roi = frame[120:420, 80:560]

    # -------------------------------------

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (5,5), 0)

    _, thresh = cv2.threshold(
        blur,
        120,
        255,
        cv2.THRESH_BINARY_INV
    )

    # ---------------- CONTOURS ----------------

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # ------------------------------------------

    current_objects = {}

    for cnt in contours:

        area = cv2.contourArea(cnt)

        if area > 1000:

            # Perimeter

            peri = cv2.arcLength(cnt, True)

            # Polygon approximation

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

            matched = False

            current_id = -1

            # ---------------- TRACKING ----------------

            for id, obj in tracked_objects.items():

                px, py = obj["center"]

                distance = math.sqrt(
                    (cx - px)**2 +
                    (cy - py)**2
                )

                if distance < 40:

                    current_objects[id] = {

                        "center": (cx, cy),
                        "shape": shape,
                        "counted": obj["counted"]
                    }

                    current_id = id

                    matched = True

                    # ---------------- LINE CROSSING ----------------

                    if (
                        py < line_y and
                        cy >= line_y and
                        obj["counted"] == False
                    ):

                        if shape in counts:

                            counts[shape] += 1

                        current_objects[id]["counted"] = True

                    break

            # ---------------- NEW OBJECT ----------------

            if matched == False:

                object_id += 1

                current_id = object_id

                current_objects[current_id] = {

                    "center": (cx, cy),
                    "shape": shape,
                    "counted": False

                }

            # ------------------------------------------------

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
                (x+w, y+h),
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

            # Display shape + ID

            cv2.putText(
                roi,
                f"{shape} ID:{current_id}",
                (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255,255,0),
                2
            )

    # ---------------- UPDATE TRACKER ----------------

    tracked_objects = current_objects

    # ------------------------------------------------

    # Draw counting line

    cv2.line(
        roi,
        (0, line_y),
        (480, line_y),
        (0,255,255),
        2
    )

    # ---------------- DISPLAY COUNTS ----------------

    y_offset = 30

    for shape_name, count in counts.items():

        cv2.putText(
            frame,
            f"{shape_name}: {count}",
            (20, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0,0,255),
            2
        )

        y_offset += 30

    # ------------------------------------------------

    # Draw ROI rectangle

    cv2.rectangle(
        frame,
        (80,120),
        (560,420),
        (255,0,0),
        2
    )

    # Show windows

    cv2.imshow("ROI", roi)

    cv2.imshow("Threshold", thresh)

    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()