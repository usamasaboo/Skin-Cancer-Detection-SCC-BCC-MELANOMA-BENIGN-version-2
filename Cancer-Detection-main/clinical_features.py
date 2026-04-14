import cv2
import numpy as np
from PIL import Image

def get_lesion_mask(img_bgr):
    """Simple thresholding/contour-based mask extraction."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # Otsu's thresholding
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Morphological cleanup
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Filter by area and solidity to remove hair/noise
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_mask = np.zeros_like(mask)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 300: # Remove tiny noise
            continue
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0: continue
        solidity = area / float(hull_area)
        
        # Hair/noise is usually very spider-web-like (low solidity)
        if solidity > 0.4:
            cv2.drawContours(filtered_mask, [cnt], -1, 255, -1)
            
    return filtered_mask

def calculate_asymmetry(mask):
    """Calculates asymmetry by flipping the mask."""
    if np.sum(mask > 0) == 0: return 0.0
    h, w = mask.shape
    h_flip = cv2.flip(mask, 1)
    diff_h = cv2.bitwise_xor(mask, h_flip)
    asym_h = np.sum(diff_h > 0) / np.sum(mask > 0)
    
    v_flip = cv2.flip(mask, 0)
    diff_v = cv2.bitwise_xor(mask, v_flip)
    asym_v = np.sum(diff_v > 0) / np.sum(mask > 0)
    
    return min((asym_h + asym_v) / 2.0, 1.0)

def calculate_border(mask):
    """Compactness index: P^2 / (4 * pi * A)"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return 0.0
    
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    
    if area == 0: return 0.0
    
    compactness = (perimeter ** 2) / (4 * np.pi * area)
    return min(max((compactness - 1.0) / 4.0, 0), 1.0)

def detect_shiny_regions(img_bgr, mask):
    """BCC Rule: Pink + shiny. Detect localized high-intensity reflections relative to surrounding skin."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    lesion_pixels = gray[mask > 0]
    if lesion_pixels.size == 0: return 0.0
    
    # Adaptive glare: compare against background skin tone, not just absolute brightness
    bg_mask = cv2.bitwise_not(mask)
    bg_pixels = gray[bg_mask > 0]
    
    # If the image is mostly mask, fallback to simple calculation
    if bg_pixels.size < 1000:
        bg_mean = np.mean(gray)
    else:
        bg_mean = np.mean(bg_pixels)
        
    # Find pixels that are significantly brighter than the background mean
    # A true shiny BCC nodule is much brighter than surrounding skin. 
    # Glare on white skin is bright, but the background is also quite bright.
    threshold = max(bg_mean + 40, np.percentile(lesion_pixels, 92))
    
    shiny_mask = (gray > threshold) & (mask > 0)
    
    shiny_score = np.sum(shiny_mask) / np.sum(mask > 0)
    return min(shiny_score * 8.0, 1.0)

def calculate_roughness(img_bgr, mask):
    """SCC Rule: Rough + Crusted. Higher sensitivity Laplacian + Edge density."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # Higher sensitivity to localized texture
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    masked_lap = np.abs(laplacian[mask > 0])
    if masked_lap.size == 0: return 0
    
    # Texture variance
    tex_score = np.mean(masked_lap) / 30.0
    
    # Edge density (scaly surface)
    edges = cv2.Canny(gray, 30, 100)
    edge_density = np.sum(edges[mask > 0] > 0) / np.sum(mask > 0)
    
    return min(tex_score + edge_density * 3.0, 1.0)

def detect_redness(img_bgr, mask):
    """SCC Rule: Red/Reddish-brown inflammation. Increased sensitivity."""
    if np.sum(mask > 0) == 0: return 0
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    a_lesion = a[mask > 0].astype(np.float32)
    # 128 is neutral. SCC often has a_channel > 140
    red_score = (np.mean(a_lesion) - 132) / 15.0 
    return min(max(red_score, 0), 1.0)

def detect_ulceration(img_bgr, mask):
    """Malignancy marker: Detect localized intensity dips/crust."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    lesion_pixels = gray[mask > 0]
    if lesion_pixels.size == 0: return 0
    
    mean_val = np.mean(lesion_pixels)
    dark_mask = (gray < (mean_val * 0.7)) & (mask > 0)
    
    ulcer_score = np.sum(dark_mask) / np.sum(mask > 0)
    return min(ulcer_score * 12.0, 1.0)

def detect_multicolor(img_bgr, mask):
    """Melanoma Rule: Multi-colored clusters."""
    if np.sum(mask > 0) == 0: return 0
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    s_std = np.std(s[mask > 0])
    v_std = np.std(v[mask > 0])
    
    color_score = (s_std / 35.0 + v_std / 35.0) / 2.0
    return min(color_score, 1.0)

def is_skin_specimen(img_bgr, mask):
    """
    Enhanced Validation Layer: Detects if the image is likely a skin specimen.
    Focus: Rejects "Neutral" chaos (fog/clouds) and accepts "Healthy" uniformity (hands).
    """
    img_area = img_bgr.shape[0] * img_bgr.shape[1]
    
    # 1. COLOR & SATURATION VALIDATION
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    # --- SATURATION/NEUTRALITY CHECK (Fog/Cloud Filter) ---
    # Human skin (even pale) has a minimum chroma. Fog/Clouds are near-zero saturation.
    avg_sat = np.mean(s)
    if avg_sat < 18: # Absolute neutrality floor
        return False

    # Skin color masks (Broadened for varied lighting)
    skin_hue_mask = cv2.inRange(h, 0, 30) | cv2.inRange(h, 160, 180)
    skin_sat_mask = cv2.inRange(s, 20, 180) 
    skin_color_hsv = skin_hue_mask & skin_sat_mask
    
    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    cr_mask = cv2.inRange(cr, 131, 175) # Slightly wider window
    cb_mask = cv2.inRange(cb, 75, 129)
    skin_color_ycrcb = cr_mask & cb_mask
    
    # Validation Synthesis
    combined_skin = (skin_color_hsv & skin_color_ycrcb)
    skin_ratio = np.sum(combined_skin > 0) / img_area
    ycrcb_only_ratio = np.sum(skin_color_ycrcb > 0) / img_area
    
    # 2. SKY / BLUE DOMINANCE REJECTION
    blue_mask = cv2.inRange(h, 85, 135) & cv2.inRange(s, 40, 255)
    blue_ratio = np.sum(blue_mask > 0) / img_area
    if blue_ratio > 0.35: 
        return False

    # 3. TEXTURE & SURFACE ANALYSIS
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    std_dev = np.std(gray)
    
    # Edge density
    edges = cv2.Canny(gray, 35, 100)
    edge_density = np.sum(edges > 0) / img_area
    
    # --- ADAPTIVE UNIFORMITY (Hand/Body Skin Fix) ---
    # Healthy skin is uniform but not "flat". 
    # Fog is uniform but lack saturation. 
    # We accept high uniformity (low std_dev) ONLY if saturation is skin-like.
    is_skin_like_uniform = std_dev < 20.0 and ycrcb_only_ratio > 0.7
    
    # Rejection Cases
    # Too Uniform + Not skin-saturated = Wall/Sky/Fog
    if std_dev < 10.0 and ycrcb_only_ratio < 0.3:
        return False
        
    # Too Glossy (Car)
    ret, high_val = cv2.threshold(v, 240, 255, cv2.THRESH_BINARY)
    gloss_ratio = np.sum(high_val > 0) / img_area
    if gloss_ratio > 0.20 and edge_density < 0.008:
        return False

    # 5. FINAL DECISION
    # Accept if: 
    # A) Strong color match + reasonable context
    # B) Clean uniform skin (like a hand)
    is_legit_skin = (skin_ratio > 0.3) or (ycrcb_only_ratio > 0.6 and avg_sat > 25)
    
    # Minimum vital signs of a clinical image
    has_image_content = edge_density > 0.001 or is_skin_like_uniform
    
    return bool(is_legit_skin and has_image_content)

def is_blank_skin(img_bgr, mask):
    """Detect if the image is just plain skin without significant marks."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_area = img_bgr.shape[0] * img_bgr.shape[1]
    
    # 1. Edge Density Check (Great for removing plain dark skin / shadows)
    # Plain skin, even if shadowed, lacks sharp intra-lesion transitions
    edges = cv2.Canny(gray, 20, 80)
    edge_density = np.sum(edges > 0) / img_area
    if edge_density < 0.005: # Extremely smooth skin
        return True

    # 2. HSV Variance Check (Hue uniformity)
    # Plain skin usually has a very narrow band of hue, regardless of lightness
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    h_std = np.std(h)
    
    if h_std < 5.0 and np.std(gray) < 20.0:
        return True

    # 3. Mask ratio check
    mask_area = np.sum(mask > 0)
    mask_ratio = mask_area / img_area
    
    # If Otsu thresholding grabs >75% of the image, it's usually just a lighting gradient on plain skin.
    # If it's <2%, it failed to find any lesion.
    if mask_ratio > 0.75 or mask_ratio < 0.02:
        return True
        
    return False

def extract_clinical_features(img_np_bgr):
    """MedVision Differential Diagnosis Module - Enhanced SCC Sensitivity."""
    mask = get_lesion_mask(img_np_bgr)
    
    is_skin = is_skin_specimen(img_np_bgr, mask)
    blank = is_blank_skin(img_np_bgr, mask)
    
    asym = calculate_asymmetry(mask)
    border = calculate_border(mask)
    shiny = detect_shiny_regions(img_np_bgr, mask)
    roughness = calculate_roughness(img_np_bgr, mask)
    redness = detect_redness(img_np_bgr, mask)
    ulcer = detect_ulceration(img_np_bgr, mask)
    multicolor = detect_multicolor(img_np_bgr, mask)
    
    return {
        'asymmetry': float(asym),
        'border': float(border),
        'shiny': float(shiny),
        'roughness': float(roughness),
        'redness': float(redness),
        'ulcer': float(ulcer),
        'multicolor': float(multicolor),
        'is_blank': bool(blank),
        'is_skin': bool(is_skin)
    }

if __name__ == "__main__":
    # Test
    dummy = np.zeros((224,224,3), dtype=np.uint8) + 150
    cv2.circle(dummy, (112, 112), 50, (40, 50, 180), -1) # Reddish-brown SCC-like
    print(extract_clinical_features(dummy))
