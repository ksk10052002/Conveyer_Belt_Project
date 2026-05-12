import cv2
import numpy as np
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# Open file picker
Tk().withdraw()
file_path = askopenfilename(title="Select an Image")

#Read image
image = cv2.imread(file_path)

#Safety check
if image is None:
    print("Error: Image not loaded")
    exit()
else:
    print("Image loaded successfully")

# Show original image
cv2.imshow("Input Image", image)

# ---------------- WATERSHED START ----------------

# Step 1: Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Step 2: Threshold
_, thresh = cv2.threshold(gray, 0, 255,
                          cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Step 3: Noise removal
kernel = np.ones((3,3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

# Step 4: Sure background
sure_bg = cv2.dilate(opening, kernel, iterations=3)

# Step 5: Distance transform
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)

# Step 6: Sure foreground
_, sure_fg = cv2.threshold(dist_transform,
                           0.5 * dist_transform.max(), 255, 0)

# Step 7: Unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)

# Step 8: Marker labeling
_, markers = cv2.connectedComponents(sure_fg)
markers = markers + 1
markers[unknown == 255] = 0

# Step 9: Apply watershed
markers = cv2.watershed(image, markers)

# Step 10: Mark boundaries
image[markers == -1] = [0, 0, 255]

# ---------------- WATERSHED END ----------------

# Show results
cv2.imshow("Threshold", thresh)
cv2.imshow("Watershed Result", image)

cv2.waitKey(0)
cv2.destroyAllWindows()