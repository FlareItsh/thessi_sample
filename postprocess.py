import cv2
import numpy as np

def validate_features(mask, min_circularity=0.6, min_solidity=0.8, min_area=10, max_area=2000):
    """
    Filters detected contours by circularity, solidity, and area.
    Discard objects that are too elongated (likely hairs) or too large (uneven skin).
    """
    # Ensure mask is binary (0 or 255)
    if mask.max() == 1:
        mask = (mask * 255).astype(np.uint8)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    filtered_mask = np.zeros_like(mask)
    valid_contours = []
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue
            
        # Perimeter
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
            
        # Circularity: 4 * pi * Area / (Perimeter^2)
        circularity = (4 * np.pi * area) / (perimeter ** 2)
        
        # Solidity: Area / Convex Hull Area
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / hull_area if hull_area > 0 else 0
        
        # Validation
        if circularity >= min_circularity and solidity >= min_solidity:
            valid_contours.append(cnt)
            cv2.drawContours(filtered_mask, [cnt], -1, 255, -1)
            
    return filtered_mask, valid_contours

def highlight_acne(original_img, mask):
    """
    Overlays detected acne spots with green circles on the original image.
    """
    output = original_img.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        # Get bounding circle
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        center = (int(x), int(y))
        radius = int(radius)
        
        # Draw circle
        cv2.circle(output, center, radius, (0, 255, 0), 2)
        
    return output

if __name__ == "__main__":
    # Test with dummy data
    dummy_mask = np.zeros((256, 256), dtype=np.uint8)
    cv2.circle(dummy_mask, (100, 100), 10, 255, -1) # Valid
    cv2.rectangle(dummy_mask, (50, 50), (150, 60), 255, -1) # Invalid (elongated)
    
    filtered, _ = validate_features(dummy_mask)
    cv2.imwrite("test_filtered.png", filtered)
    print("Post-processing test complete.")
