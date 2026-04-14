import cv2
import numpy as np

def test_blank_detection(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return "Image not found"
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Otsu thresholding
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    ret, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    max_area = 0
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        max_area = cv2.contourArea(cnt)
        
    img_area = gray.shape[0] * gray.shape[1]
    
    std_dev = np.std(gray)
    edges = cv2.Canny(gray, 30, 100)
    edge_density = np.sum(edges > 0) / img_area
    
    print(f"File: {img_path}")
    print(f"Max Contour Area Ratio: {max_area / img_area:.3f}")
    print(f"Std Dev: {std_dev:.2f}")
    print(f"Edge Density: {edge_density:.4f}")
    print("-" * 30)

# Create a dummy blank image (plain color)
blank_img = np.ones((224, 224, 3), dtype=np.uint8) * 150
cv2.imwrite("dummy_blank.jpg", blank_img)

# Create a dummy blank image with some noise (hair)
noise_img = np.ones((224, 224, 3), dtype=np.uint8) * 150
noise = np.random.normal(0, 10, (224, 224, 3)).astype(np.uint8)
noise_img = cv2.add(noise_img, noise)
for i in range(10): # draw some "hairs"
    cv2.line(noise_img, (np.random.randint(0, 224), np.random.randint(0, 224)), 
             (np.random.randint(0, 224), np.random.randint(0, 224)), (120, 120, 120), 1)
cv2.imwrite("dummy_hair.jpg", noise_img)

# Create dummy lesion
lesion_img = np.ones((224, 224, 3), dtype=np.uint8) * 180
cv2.circle(lesion_img, (112, 112), 40, (50, 60, 100), -1)
cv2.imwrite("dummy_lesion.jpg", lesion_img)

test_blank_detection("dummy_blank.jpg")
test_blank_detection("dummy_hair.jpg")
test_blank_detection("dummy_lesion.jpg")
