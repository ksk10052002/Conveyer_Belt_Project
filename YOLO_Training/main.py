import cv2
import numpy as np
from ultralytics import YOLO
import math

# =========================
# LOAD YOLO MODEL
# =========================
model = YOLO("yolov8n.pt")  # replace with custom trained model later

# =========================
# CAMERA
# =========================
cap = cv2.VideoCapture(0)

# =========================
# ROI (IMPORTANT - CONVEYOR AREA ONLY)
# =========================
ROI_X1, ROI_Y1 = 100, 150
ROI_X2, ROI_Y2 = 700, 500

# =========================
# COUNTING SYSTEM
# =========================
counted_ids = set()
object_id = 0
tracked_objects = {}

LINE_Y = 300  # counting line inside ROI

# =========================
# BACKGROUND SUBTRACTOR (MOTION FILTER)
# =========================
fgbg = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50)

# =========================
# HELPER FUNCTIONS
# =========================
def get_center(x1, y1, x2, y2):
    return (int((x1 + x2) / 2), int((y1 + y2) / 2))

def distance(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

# =========================
# SHAPE DETECTION
# =========================
def detect_shape(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(blur, 60, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    shape = "Unknown"

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 500:
            continue

        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)

        if len(approx) == 3:
            shape = "Triangle"
        elif len(approx) == 4:
            shape = "Rectangle"
        elif len(approx) > 6:
            shape = "Circle"
        else:
            shape = "Irregular"

    return shape

# =========================
# MAIN LOOP
# =========================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (800, 600))

    # DRAW ROI
    cv2.rectangle(frame, (ROI_X1, ROI_Y1), (ROI_X2, ROI_Y2), (255, 0, 0), 2)

    roi = frame[ROI_Y1:ROI_Y2, ROI_X1:ROI_X2]

    # MOTION MASK
    mask = fgbg.apply(roi)
    motion_score = np.sum(mask)

    # SKIP IF NO MOVEMENT (REDUCES FALSE DETECTION)
    if motion_score < 8000:
        cv2.imshow("Conveyor", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    # YOLO DETECTION ON ROI ONLY
    results = model(roi)

    detections = results[0].boxes.data.cpu().numpy()

    new_tracked = {}

    for det in detections:
        x1, y1, x2, y2, conf, cls = det

        if conf < 0.5:
            continue

        class_name = model.names[int(cls)]

        # =========================
        # BLOCK HANDS / PERSONS
        # =========================
        if class_name in ["person"]:
            continue

        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        center = get_center(x1, y1, x2, y2)

        # TRACKING
        assigned_id = None

        for tid, tcenter in tracked_objects.items():
            if distance(center, tcenter) < 60:
                assigned_id = tid
                break

        if assigned_id is None:
            assigned_id = object_id
            object_id += 1

        new_tracked[assigned_id] = center

        # CROPPED OBJECT FOR SHAPE
        crop = roi[y1:y2, x1:x2]
        shape = "Unknown"

        if crop.size > 0:
            shape = detect_shape(crop)

        # COUNTING LOGIC
        if assigned_id not in counted_ids:
            if center[1] > LINE_Y:
                counted_ids.add(assigned_id)

        # DRAW
        label = f"ID:{assigned_id} {class_name} {shape}"

        cv2.rectangle(roi, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(roi, label, (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)

    tracked_objects = new_tracked

    # SHOW COUNT
    cv2.putText(frame, f"COUNT: {len(counted_ids)}",
                (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0,0,255), 3)

    # SHOW LINE INSIDE ROI
    cv2.line(frame, (ROI_X1, ROI_Y1 + 150),
             (ROI_X2, ROI_Y1 + 150), (0,255,255), 2)

    cv2.imshow("Conveyor Belt System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()