import cv2
import numpy as np
import os

PATH = os.getcwd()

# input directory containing the images to label
IMAGE_DIR = ''
PREFIX = ''

# output directories for labeled images and YOLO label txt files
OUTPUT_IMAGE_DIR = ''
OUTPUT_LABEL_DIR = ''
os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)
os.makedirs(OUTPUT_LABEL_DIR, exist_ok=True)

# set None to process all images, or specify a single image for auditing
AUDIT_IMAGE = ''  # e.g. 'frame_0000.jpg' or None
AUDIT_IMAGE_DIR = PATH + ''

# danger zone colour
lower_magenta = np.array([140, 80, 40])
upper_magenta = np.array([170, 255, 255])

# find image files
image_files = [f for f in os.listdir(IMAGE_DIR) if f.endswith(('.jpg', '.png', '.jpeg'))]
print(f"Found {len(image_files)} images to automatically label.")

# loop through images and process each one
for img_name in image_files:
    # skip if we're auditing and this isn't the audit image
    if AUDIT_IMAGE is not None and img_name != AUDIT_IMAGE:
        continue

    # read the image
    img_path = os.path.join(IMAGE_DIR, img_name)
    frame = cv2.imread(img_path)
    if frame is None:
        continue
        
    img_h, img_w, _ = frame.shape
    
    # soften noise with median blur and convert to HSV for color segmentation
    blurred = cv2.medianBlur(frame, 3)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    mask_bullets = cv2.inRange(hsv, lower_magenta, upper_magenta)

    # morphological closing to connect nearby contours and reduce noise
    kernel = np.ones((5, 5), np.uint8)
    clean_mask = cv2.morphologyEx(mask_bullets, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # create corresponding txt file for YOLO labels
    txt_name = os.path.splitext(img_name)[0] + ".txt"
    txt_path = os.path.join(OUTPUT_LABEL_DIR, txt_name)

    yolo_labels = []  # store YOLO labels for visualization later
    is_frame_messy = False  # flag to skip saving labels for messy frames with too much noise or invalid contours

    # open the txt file for writing YOLO labels
    for i, contour in enumerate(contours):
        # filter out small contours that are likely noise
        if cv2.contourArea(contour) < 300:
            continue
        
        # smooth the contour
        epsilon = np.maximum(0.002 * cv2.arcLength(contour, True), 5)
        smoothed_contour = cv2.approxPolyDP(contour, epsilon, True)

        # discard images with contours with too many points that are likely noise
        if len(smoothed_contour) > 60:
            is_frame_messy = True
            break
        
        # flatten coords from the smoothed contour
        points = smoothed_contour.reshape(-1, 2)
        
        # store in YOLO format
        label_str = "0"
        for point in points:
            x, y = point
            label_str += f" {x/img_w:.6f} {y/img_h:.6f}"

        yolo_labels.append(label_str)
        
    if not is_frame_messy and AUDIT_IMAGE is None:
        # Create new unique filenames
        new_base_name = f"{PREFIX}_{os.path.splitext(img_name)[0]}"
        new_img_name = new_base_name + os.path.splitext(img_name)[1]
        new_txt_name = new_base_name + ".txt"
        
        # Target paths in the master folder
        final_img_path = os.path.join(OUTPUT_IMAGE_DIR, new_img_name)
        final_txt_path = os.path.join(OUTPUT_LABEL_DIR, new_txt_name)
        
        # Save the label file
        with open(final_txt_path, "w") as f:
            for label in yolo_labels:
                f.write(label + "\n")
                
        # Copy/Save the image to the master image folder
        cv2.imwrite(final_img_path, frame)

    # visualize result of one image
    if img_name == AUDIT_IMAGE:
        if not is_frame_messy:
            for label in yolo_labels:
                parts = label.split()
                class_id = int(parts[0])
                coords = np.array([float(x) for x in parts[1:]])
                
                # reshape back to pixel coords
                pts_x = (coords[0::2] * img_w).astype(np.int32)
                pts_y = (coords[1::2] * img_h).astype(np.int32)
                
                # reconstruct contour
                reconstructed_contour = np.stack((pts_x, pts_y), axis=1).reshape(-1, 1, 2)

                # draw contour
                cv2.drawContours(frame, [reconstructed_contour], -1, (0, 0, 255), 3)

        # save a snapshot to see if green vs blue lines up correctly with the game reality
        cv2.imwrite(AUDIT_IMAGE_DIR + "/audit_test.jpg", frame)
        print(f"Audited {AUDIT_IMAGE} and saved visualization.")

        break  # stop after processing the audit image

print("Labeling complete")