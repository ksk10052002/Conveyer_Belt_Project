import cv2
import math

cap = cv2.VideoCapture(0)

# ---------------- TRACKER ----------------

tracked_objects = {}

object_id = 0

# -----------------------------------------

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # ---------------- ROI ----------------

    roi = frame[120:420, 80:560]

    # -------------------------------------

    # Grayscale

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Blur

    blur = cv2.GaussianBlur(gray, (5,5), 0)

    # Threshold

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

    # Current frame centers

    current_centers = []

    for cnt in contours:

        area = cv2.contourArea(cnt)

        if area > 1000:

            # Bounding box

            x, y, w, h = cv2.boundingRect(cnt)

            # Center point

            cx = int(x + w/2)
            cy = int(y + h/2)

            current_centers.append((cx, cy))

            # Draw rectangle

            cv2.rectangle(roi, (x,y), (x+w, y+h), (255,0,0), 2)


            # Draw center

            cv2.circle(
                roi,
                (cx, cy),
                5,
                (0,0,255),
                -1
            )

    # ---------------- TRACKING ----------------

    new_tracked_objects = {}

    for center in current_centers:

        cx, cy = center

        same_object_detected = False

        for id, pt in tracked_objects.items():

            px, py = pt

            # Euclidean distance

            distance = math.sqrt(
                (cx - px)**2 +
                (cy - py)**2
            )

            # Match object

            if distance < 30:

                new_tracked_objects[id] = (cx, cy)

                same_object_detected = True

                break

        # New object

        if same_object_detected == False:

            object_id += 1

            new_tracked_objects[object_id] = (cx, cy)

    # Update tracker

    tracked_objects = new_tracked_objects

    # ------------------------------------------

    # Display IDs

    for id, pt in tracked_objects.items():

        cx, cy = pt

        cv2.putText(
            roi,
            f"ID:{id}",
            (cx-20, cy-20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255,255,0),
            2
        )

    cv2.rectangle(
        frame,
        (80,120),
        (560,420),
        (255,0,0),
        2
    )

    # Show windows

    cv2.imshow("Full Frame", frame)


    cv2.imshow("ROI", roi)

    cv2.imshow("Threshold", thresh)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()