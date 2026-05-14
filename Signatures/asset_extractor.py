import os
import cv2
import numpy as np
import shutil
from sklearn.cluster import KMeans
from glob import glob

# --- Configuration ---
DATASET_IMAGES_DIR = "target_dataset/images" # Update this to your dataset images path
DATASET_LABELS_DIR = "target_dataset/labels" # Update this to your dataset labels path
OUTPUT_DIR = "processed_assets"
DIR_COLORED = os.path.join(OUTPUT_DIR, "clean_colored")
DIR_BLACK = os.path.join(OUTPUT_DIR, "clean_black")
DIR_DISCARDED = os.path.join(OUTPUT_DIR, "discarded_crops")

# High density threshold for rejecting black ink (e.g., 15% black pixels means heavy text overlap)
BLACK_DENSITY_THRESHOLD = 0.15 

def setup_directories():
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(DIR_COLORED, exist_ok=True)
    os.makedirs(DIR_BLACK, exist_ok=True)
    os.makedirs(DIR_DISCARDED, exist_ok=True)

def yolo_to_xyxy(x_center, y_center, width, height, img_w, img_h):
    x1 = int((x_center - width / 2) * img_w)
    y1 = int((y_center - height / 2) * img_h)
    x2 = int((x_center + width / 2) * img_w)
    y2 = int((y_center + height / 2) * img_h)
    return max(0, x1), max(0, y1), min(img_w, x2), min(img_h, y2)

def detect_ink_color(crop_rgb):
    """
    Uses K-Means to find dominant colors and identifies if ink is colored or black.
    """
    # Flatten the image array
    pixels = crop_rgb.reshape(-1, 3)
    
    # Use KMeans to cluster into 2 colors (Paper vs Ink)
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    kmeans.fit(pixels)
    
    colors = kmeans.cluster_centers_
    # Find the ink cluster (the darker one usually, smaller sum of RGB)
    sums = colors.sum(axis=1)
    ink_color = colors[np.argmin(sums)]
    
    r, g, b = ink_color
    
    # Simple heuristic to distinguish black/gray from colored ink
    # If the variance between RGB channels is very low, it's grayscale (black ink)
    if np.var([r, g, b]) < 150 and sum([r,g,b]) < 300:
        return "black", ink_color
    return "colored", ink_color

def process_colored_ink(crop_bgr):
    """
    Applies dynamic HSV masking and morphological ops for colored stamps/signatures.
    """
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    
    # We use a broad mask for any non-grayscale/non-white ink.
    # Exclude very low saturation (black/gray) and very high value (white paper)
    lower_bound = np.array([0, 30, 0])
    upper_bound = np.array([179, 255, 220])
    
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    
    # Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Gaussian blur for anti-aliasing
    mask = cv2.GaussianBlur(mask, (3, 3), 0)
    
    # Create RGBA
    b, g, r = cv2.split(crop_bgr)
    rgba = cv2.merge((b, g, r, mask))
    
    # Crop to bounding rect of the mask to remove excess padding
    coords = cv2.findNonZero(mask)
    if coords is not None:
        y, x = coords[:, 0, 0], coords[:, 0, 1]
        rgba = rgba[min(y):max(y)+1, min(x):max(x)+1]
        
    return rgba

def process_black_ink(crop_bgr):
    """
    Grayscale analysis, adaptive thresholding, and density check for black ink.
    Returns (rgba_image, None) if accepted, or (None, crop_bgr) if rejected.
    """
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    
    # Adaptive thresholding to segment ink from paper
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
    
    # Calculate black pixel density
    total_pixels = thresh.shape[0] * thresh.shape[1]
    black_pixels = np.count_nonzero(thresh)
    density = black_pixels / total_pixels
    
    if density > BLACK_DENSITY_THRESHOLD:
        # Reject: likely overlapped with heavy background printed text
        return None, crop_bgr
    
    # Create transparent PNG
    mask = thresh.copy()
    mask = cv2.GaussianBlur(mask, (3, 3), 0) # anti-aliasing
    
    b, g, r = cv2.split(crop_bgr)
    rgba = cv2.merge((b, g, r, mask))
    
    # Tight crop
    coords = cv2.findNonZero(mask)
    if coords is not None:
        y, x = coords[:, 0, 0], coords[:, 0, 1]
        rgba = rgba[min(y):max(y)+1, min(x):max(x)+1]
        return rgba, None
        
    return rgba, None

def main():
    setup_directories()
    
    image_files = glob(os.path.join(DATASET_IMAGES_DIR, "*.jpg")) + glob(os.path.join(DATASET_IMAGES_DIR, "*.png"))
    
    if not image_files:
        print(f"No images found in {DATASET_IMAGES_DIR}. Please configure your path.")
        return
        
    print(f"Found {len(image_files)} images. Starting extraction...")
    
    crop_counter = 0
    discarded_counter = 0
    colored_counter = 0
    black_counter = 0
    
    for img_path in image_files:
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(DATASET_LABELS_DIR, base_name + ".txt")
        
        if not os.path.exists(label_path):
            continue
            
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        img_h, img_w = img.shape[:2]
        
        with open(label_path, 'r') as f:
            lines = f.readlines()
            
        for idx, line in enumerate(lines):
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id, cx, cy, w, h = map(float, parts[:5])
                x1, y1, x2, y2 = yolo_to_xyxy(cx, cy, w, h, img_w, img_h)
                
                # Verify crop validity
                if x2 - x1 < 5 or y2 - y1 < 5:
                    continue
                    
                crop_bgr = img[y1:y2, x1:x2]
                crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                
                ink_type, dominant_color = detect_ink_color(crop_rgb)
                crop_counter += 1
                
                if ink_type == "colored":
                    rgba_crop = process_colored_ink(crop_bgr)
                    if rgba_crop is not None and rgba_crop.size > 0:
                        out_file = os.path.join(DIR_COLORED, f"colored_{base_name}_{idx}.png")
                        cv2.imwrite(out_file, rgba_crop)
                        colored_counter += 1
                else:
                    rgba_crop, rejected_crop = process_black_ink(crop_bgr)
                    if rejected_crop is not None:
                        out_file = os.path.join(DIR_DISCARDED, f"rejected_{base_name}_{idx}.jpg")
                        cv2.imwrite(out_file, rejected_crop)
                        discarded_counter += 1
                    elif rgba_crop is not None and rgba_crop.size > 0:
                        out_file = os.path.join(DIR_BLACK, f"black_{base_name}_{idx}.png")
                        cv2.imwrite(out_file, rgba_crop)
                        black_counter += 1

    print("\n--- Extraction Complete ---")
    print(f"Total Crops Analyzed: {crop_counter}")
    print(f"Colored Ink Extracted: {colored_counter}")
    print(f"Clean Black Extracted: {black_counter}")
    print(f"Discarded (High Density): {discarded_counter}")
    
    print("\nCompressing assets...")
    shutil.make_archive('extracted_assets', 'zip',OUTPUT_DIR)
    print("Successfully created extracted_assets.zip.")

if __name__ == "__main__":
    main()
