import numpy as np
import os
import cv2
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions

def detect_traffic_light_color(cropped_light):
    """Identify traffic light color based on HSV filtering."""
    # Convert to HSV color space
    hsv_img = cv2.cvtColor(cropped_light, cv2.COLOR_BGR2HSV)

    # Define HSV ranges for red, yellow, and green
    red_lower1 = np.array([0, 70, 50])
    red_upper1 = np.array([10, 255, 255])
    red_lower2 = np.array([170, 70, 50])
    red_upper2 = np.array([180, 255, 255])

    yellow_lower = np.array([15, 100, 100])
    yellow_upper = np.array([35, 255, 255])

    green_lower = np.array([40, 50, 50])
    green_upper = np.array([90, 255, 255])

    # Create masks for each color
    red_mask = cv2.inRange(hsv_img, red_lower1, red_upper1) | cv2.inRange(hsv_img, red_lower2, red_upper2)
    yellow_mask = cv2.inRange(hsv_img, yellow_lower, yellow_upper)
    green_mask = cv2.inRange(hsv_img, green_lower, green_upper)

    # Count non-zero pixels for each mask
    red_pixels = cv2.countNonZero(red_mask)
    yellow_pixels = cv2.countNonZero(yellow_mask)
    green_pixels = cv2.countNonZero(green_mask)

    # Determine the dominant color
    if red_pixels > yellow_pixels and red_pixels > green_pixels:
        return "Red Light"
    elif yellow_pixels > red_pixels and yellow_pixels > green_pixels:
        return "Yellow Light"
    elif green_pixels > red_pixels and green_pixels > yellow_pixels:
        return "Green Light"
    else:
        return "Color Undetected"

# Load the pre-trained ResNet50 model
model = ResNet50(weights='imagenet')

# Directory containing traffic light images
image_dir = 'traffic_light_image'
log_file_path = 'classification_results.log'

# Open log file for writing
with open(log_file_path, 'w') as log_file:
    for file_name in os.listdir(image_dir):
        file_path = os.path.join(image_dir, file_name)

        if os.path.isfile(file_path) and file_name.lower().endswith(('jpg', 'jpeg', 'png')):
            try:
                img = cv2.imread(file_path)
                if img is None:
                    raise ValueError("Image not loaded correctly")

                # Resize for processing and detect ROI (manual or automated object detection can improve this)
                img_resized = cv2.resize(img, (224, 224))

                # Force detection on the center crop (ROI simulation)
                h, w, _ = img.shape
                center_crop = img[h // 4: 3 * h // 4, w // 4: 3 * w // 4]

                # Detect traffic light color
                color_result = detect_traffic_light_color(center_crop)

                log_file.write(f"Image: {file_name}\n")
                log_file.write(f"  Traffic Light Color: {color_result}\n\n")
                print(f"Processed {file_name} - {color_result}")

            except Exception as e:
                log_file.write(f"Error processing {file_name}: {str(e)}\n")
                print(f"Error processing {file_name}: {e}")

print("Processing complete. Results saved in 'classification_results.log'.")
