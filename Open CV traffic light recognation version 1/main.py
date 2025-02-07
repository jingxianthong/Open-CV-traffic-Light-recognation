import cv2
import numpy as np

# Load the traffic light image
image_path = "traffic_light_image/traffic_green03.jpg"  # Replace with your image path
image = cv2.imread(image_path)

# Resize for display and processing
resized_img = cv2.resize(image, (600, 600))
hsv = cv2.cvtColor(resized_img, cv2.COLOR_BGR2HSV)

# Define HSV ranges for red, yellow, and green colors
color_ranges = {
    "Red": [(0, 100, 100), (10, 255, 255)],
    "Yellow": [(15, 100, 100), (35, 255, 255)],
    "Green": [(45, 100, 100), (75, 255, 255)]
}

# Check for each color mask
for color, (lower, upper) in color_ranges.items():
    mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
    result = cv2.bitwise_and(resized_img, resized_img, mask=mask)
    
    if cv2.countNonZero(mask) > 500:  # Adjust this threshold for better results
        print(f"{color} Light Detected!")
        break

# Display the result
cv2.imshow("Traffic Light Detection", resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
