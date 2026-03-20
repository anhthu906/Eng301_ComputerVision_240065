import cv2
import os

input_folder = "/Users/stella/Documents/Eng301_ComputerVision_240065/Assignment3/input"
valid_exts = (".png", ".jpeg", ".jpg", ".bmp", ".tiff", ".webp")

for filename in os.listdir(input_folder):
    file_path = os.path.join(input_folder, filename)
    if not filename.lower().endswith(valid_exts):
        continue

    if filename.lower().endswith(".jpg"):
        continue
    img = cv2.imread(file_path)

    if img is None:
        print(f"Cannot read {filename}, skipping...")
        continue

    new_name = os.path.splitext(filename)[0] + ".jpg"
    new_path = os.path.join(input_folder, new_name)

    cv2.imwrite(new_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    os.remove(file_path)

    print(f"Converted: {filename} -> {new_name}")

print("Done converting all images!")