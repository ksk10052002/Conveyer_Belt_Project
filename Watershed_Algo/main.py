# import cv2
# import numpy as np
# from tkinter import Tk
# from tkinter.filedialog import askopenfilename

# # Open file picker
# # Tk().withdraw()
# # file_path = askopenfilename(title="Select an Image")

# #Read image
# # image = cv2.imread(file_path)
# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break


# # #Safety check
# # if image is None:
# #     print("Error: Image not loaded")
# #     exit()
# # else:
# #     print("Image loaded successfully")

# # # Show original image
# # cv2.imshow("Input Image", image)

# # ---------------- WATERSHED START ----------------

# # Step 1: Convert to grayscale
# # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     output = frame.copy()

#     # Step 2: Threshold
#     _, thresh = cv2.threshold(gray, 0, 255,
#                             cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

#     # Step 3: Noise removal
#     kernel = np.ones((3,3), np.uint8)
#     opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

#     # Step 4: Sure background
#     sure_bg = cv2.dilate(opening, kernel, iterations=3)

#     # Step 5: Distance transform
#     dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)

#     # Step 6: Sure foreground
#     _, sure_fg = cv2.threshold(dist_transform,
#                             0.5 * dist_transform.max(), 255, 0)

#     # Step 7: Unknown region
#     sure_fg = np.uint8(sure_fg)
#     unknown = cv2.subtract(sure_bg, sure_fg)

#     # Step 8: Marker labeling
#     _, markers = cv2.connectedComponents(sure_fg)
#     markers = markers + 1
#     markers[unknown == 255] = 0

#     # Step 9: Apply watershed
#     markers = cv2.watershed(output, markers)

#     # Step 10: Mark boundaries
#     output[markers == -1] = [0, 0, 255]

#     # ---------------- WATERSHED END ----------------

#     # Show results
#     # cv2.imshow("Threshold", thresh)
#     cv2.imshow("Watershed Result", output)

#     if cv2.waitKey(1) == 27:
#         break

# cap.release()
# cv2.destroyAllWindows()






import cv2
import numpy as np

cap = cv2.VideoCapture(0)

counts = {
    "Triangle": 0,
    "Square": 0,
    "Rectangle": 0,
    "Circle": 0
}

tracked_centers = []

line_y = 150  # inside ROI

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # -------- ROI --------
    roi = frame[200:400, 100:600]
    output = roi.copy()

    # -------- PREPROCESSING --------
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7,7), 0)

    thresh = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11,
        2
    )

    # -------- MORPHOLOGY --------
    kernel = np.ones((5,5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # -------- CONTOURS --------
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)

        # -------- AREA FILTER --------
        if 1000 < area < 8000:

            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

            x, y, w, h = cv2.boundingRect(approx)

            # -------- SIZE FILTER --------
            if w < 20 or h < 20:
                continue

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
                continue

            # -------- TRACKING --------
            new_object = True
            for pt in tracked_centers:
                if abs(cx - pt[0]) < 30 and abs(cy - pt[1]) < 30:
                    new_object = False
                    break

            if new_object:
                tracked_centers.append((cx, cy))

                # -------- LINE COUNT --------
                if abs(cy - line_y) < 10:
                    counts[shape] += 1

            # -------- DRAW --------
            cv2.putText(output, shape, (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

            cv2.drawContours(output, [approx], -1, (0,255,0), 2)
            cv2.circle(output, (cx, cy), 4, (0,0,255), -1)

    # -------- DRAW LINE --------
    cv2.line(output, (0, line_y), (500, line_y), (255,0,0), 2)

    # -------- DISPLAY COUNTS --------
    y_offset = 20
    for shape_name, count in counts.items():
        cv2.putText(frame, f"{shape_name}: {count}",
                    (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255,0,0), 2)
        y_offset += 25

    cv2.imshow("ROI Detection", output)
    cv2.imshow("Threshold", thresh)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()