import os
from PIL import Image

# Directory containing images
image_folder = "traffic_light_image"



# Supported formats with case insensitivity
supported_formats = ('.png', '.jpeg', '.bmp', '.gif', '.webp')

# Conversion loop
file_list = [f for f in os.listdir(image_folder) if not f.startswith(".")]

for file_name in file_list:
    input_path = os.path.join(image_folder, file_name)

    # Process only supported formats
    if file_name.lower().endswith(tuple(ext.lower() for ext in supported_formats)):
        try:
            # Open image and convert to RGB before saving
            with Image.open(input_path) as img:
                rgb_img = img.convert('RGB')
                print(f"Converted: {file_name} to JPG")
        except Exception as e:
            print(f"Error converting {file_name}: {e}")
    else:
        print(f"Skipping unsupported file: {file_name}")

print("Image conversion completed.")
