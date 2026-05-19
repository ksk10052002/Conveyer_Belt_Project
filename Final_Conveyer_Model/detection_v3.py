import cv2
import math
import numpy as np

# ---------------- CAMERA ----------------

cap = cv2.VideoCapture(0)

# Optional camera resolution

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# ---------------- TRACKING ----------------

tracked_objects = {}

object_id = 0

# ---------------- COUNTING LINE ----------------

line_y = 150

# ---------------- SHAPE COUNTS ----------------

counts = {
    "Triangle": 0,
    "Square": 0,
    "Rectangle": 0,
    "Circle": 0
}

# ======================================================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # ---------------- ROI ----------------

    roi = frame[120:420, 80:560]

    # -------------------------------------

    # -------- GRAYSCALE --------

    gray = cv2.cvtColor(
        roi,
        cv2.COLOR_BGR2GRAY
    )

    # -------- BLUR --------

    blur = cv2.GaussianBlur(
        gray,
        (3,3),
        0
    )

    # -------- THRESHOLD --------

    _, thresh = cv2.threshold(
        blur,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # -------- MORPHOLOGY --------

    kernel = np.ones((3,3), np.uint8)

    opening = cv2.morphologyEx(
        thresh,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1
    )

    # -------- SURE BACKGROUND --------

    sure_bg = cv2.dilate(
        opening,
        kernel,
        iterations=1
    )

    # -------- DISTANCE TRANSFORM --------

    dist_transform = cv2.distanceTransform(
        opening,
        cv2.DIST_L2,
        5
    )

    # -------- DISPLAY DISTANCE MAP --------

    dist_display = cv2.normalize(
        dist_transform,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    dist_display = np.uint8(dist_display)

    # -------- SURE FOREGROUND --------

    _, sure_fg = cv2.threshold(
        dist_transform,
        0.2 * dist_transform.max(),
        255,
        0
    )

    sure_fg = np.uint8(sure_fg)

    # -------- UNKNOWN REGION --------

    unknown = cv2.subtract(
        sure_bg,
        sure_fg
    )

    # -------- CONNECTED COMPONENTS --------

    _, markers = cv2.connectedComponents(
        sure_fg
    )

    markers = markers + 1

    markers[unknown == 255] = 0

    # -------- WATERSHED --------

    markers = cv2.watershed(
        roi,
        markers
    )

    # -------- DRAW BOUNDARY --------

    roi[markers == -1] = [0,0,255]

    # ======================================================

    current_objects = {}

    # -------- EXTRACT SEPARATED OBJECTS --------

    unique_labels = np.unique(markers)

    for label in unique_labels:

        # Ignore background and boundary

        if label == 1 or label == -1:
            continue

        # -------- CREATE MASK --------

        mask = np.zeros(
            gray.shape,
            dtype=np.uint8
        )

        mask[markers == label] = 255

        # -------- FIND CONTOURS --------

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        for cnt in contours:

            area = cv2.contourArea(cnt)

            # Ignore tiny noise

            if area > 500:

                # -------- POLYGON APPROX --------

                peri = cv2.arcLength(
                    cnt,
                    True
                )

                approx = cv2.approxPolyDP(
                    cnt,
                    0.02 * peri,
                    True
                )

                vertices = len(approx)

                # -------- BOUNDING BOX --------

                x, y, w, h = cv2.boundingRect(
                    approx
                )

                # -------- CENTER POINT --------

                cx = int(x + w/2)
                cy = int(y + h/2)

                # ======================================================
                # SHAPE DETECTION
                # ======================================================

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

                # ======================================================
                # TRACKING
                # ======================================================

                matched = False

                current_id = -1

                for id, obj in tracked_objects.items():

                    px, py = obj["center"]

                    distance = math.sqrt(
                        (cx - px)**2 +
                        (cy - py)**2
                    )

                    # Same object condition

                    if distance < 40:

                        current_id = id

                        matched = True

                        current_objects[id] = {

                            "center": (cx, cy),

                            "counted": obj["counted"],

                            "missed": 0
                        }

                        # -------- LINE CROSSING --------

                        if (
                            py < line_y and
                            cy >= line_y and
                            obj["counted"] == False
                        ):

                            if shape in counts:

                                counts[shape] += 1

                            current_objects[id]["counted"] = True

                        break

                # -------- NEW OBJECT --------

                if matched == False:

                    object_id += 1

                    current_id = object_id

                    current_objects[current_id] = {

                        "center": (cx, cy),

                        "counted": False,

                        "missed": 0
                    }

                # ======================================================
                # DRAWING
                # ======================================================

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

                # Draw center point

                cv2.circle(
                    roi,
                    (cx, cy),
                    5,
                    (0,0,255),
                    -1
                )

                # Draw shape + ID

                cv2.putText(
                    roi,
                    f"{shape} ID:{current_id}",
                    (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255,255,0),
                    2
                )
    # -------- HANDLE MISSING OBJECTS --------

    for id, obj in tracked_objects.items():

        if id not in current_objects:

            obj["missed"] += 1

            # Keep object temporarily

            if obj["missed"] < 8:

                current_objects[id] = obj
            
    # ======================================================
    # UPDATE TRACKER
    # ======================================================

    tracked_objects = current_objects

    # ======================================================
    # DRAW COUNTING LINE
    # ======================================================

    cv2.line(
        roi,
        (0, line_y),
        (480, line_y),
        (0,255,255),
        2
    )

    # ======================================================
    # DISPLAY COUNTS
    # ======================================================

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

    # ======================================================
    # DRAW ROI RECTANGLE
    # ======================================================

    cv2.rectangle(
        frame,
        (80,120),
        (560,420),
        (255,0,0),
        2
    )

    # ======================================================
    # SHOW WINDOWS
    # ======================================================

    cv2.imshow("ROI", roi)

    cv2.imshow("Threshold", thresh)

    cv2.imshow("Opening", opening)

    cv2.imshow("Distance Transform", dist_display)

    cv2.imshow("Sure FG", sure_fg)

    cv2.imshow("Unknown Region", unknown)

    cv2.imshow("Full Frame", frame)

    # ======================================================

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ======================================================

cap.release()

cv2.destroyAllWindows()