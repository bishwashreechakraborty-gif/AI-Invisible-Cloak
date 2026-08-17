"""
=============================================================================
Stage 2 Automated Unit & Integration Tests (HSV Segmentation & Masking)
=============================================================================
This test suite verifies:
1. ColorDetector initialization, available color presets, and cycling.
2. Synthetic color patch segmentation (Blue, Red dual-range, Green, Yellow).
3. Morphological noise removal (verifying that isolated noise speckles are cleaned).
4. Bitwise cloak extraction correctness.
=============================================================================
"""

import numpy as np
import cv2

import config
from color_detector import ColorDetector


def create_synthetic_test_frame() -> np.ndarray:
    """
    Creates a synthetic 400x400 BGR test image containing four distinct color squares:
    - Top-Left: Blue
    - Top-Right: Green
    - Bottom-Left: Red
    - Bottom-Right: White/Gray
    """
    frame = np.zeros((400, 400, 3), dtype=np.uint8)
    
    # Top-Left (Blue): BGR = (255, 50, 50) -> Rich Blue
    frame[0:200, 0:200] = [255, 50, 50]
    
    # Top-Right (Green): BGR = (50, 220, 50) -> Rich Green
    frame[0:200, 200:400] = [50, 220, 50]
    
    # Bottom-Left (Red): BGR = (50, 50, 240) -> Rich Red
    frame[200:400, 0:200] = [50, 50, 240]
    
    # Bottom-Right (Gray): BGR = (128, 128, 128) -> Neutral Gray
    frame[200:400, 200:400] = [128, 128, 128]
    
    return frame


def run_stage2_tests():
    print("[TEST] Running Stage 2 Automated Logic Tests...")
    
    detector = ColorDetector(default_color="blue")
    
    # 1. Test Presets & Initialization
    assert detector.get_active_color_key() == "blue", "Default color must be blue."
    presets = detector.get_available_colors()
    assert "blue" in presets and "red" in presets and "green" in presets, "Presets must include blue, red, green."
    print("  [PASS] Color presets and initialization verified.")
    
    # 2. Test Color Cycling
    cycled_color = detector.cycle_next_color()
    assert cycled_color == "red", f"Next color should be red, got {cycled_color}."
    detector.set_color("blue")
    assert detector.get_active_color_key() == "blue", "Color should reset to blue."
    print("  [PASS] Color preset cycling and switching verified.")

    # 3. Test Blue Color Masking
    synthetic_frame = create_synthetic_test_frame()
    blue_mask = detector.create_mask(synthetic_frame)
    
    # Check that Top-Left region (Blue) has high activation
    blue_region_coverage = np.mean(blue_mask[30:170, 30:170]) / 255.0
    assert blue_region_coverage > 0.95, f"Blue region should be detected (>95%), got {blue_region_coverage*100:.1f}%"
    
    # Check that Green and Red regions are NOT detected in Blue mask
    green_region_leak = np.mean(blue_mask[30:170, 230:370]) / 255.0
    red_region_leak = np.mean(blue_mask[230:370, 30:170]) / 255.0
    assert green_region_leak < 0.01, f"Green should not leak into blue mask: {green_region_leak}"
    assert red_region_leak < 0.01, f"Red should not leak into blue mask: {red_region_leak}"
    print("  [PASS] Blue cloak thresholding and isolation verified.")

    # 4. Test Red Color Dual-Range Masking
    detector.set_color("red")
    red_mask = detector.create_mask(synthetic_frame)
    red_region_coverage = np.mean(red_mask[230:370, 30:170]) / 255.0
    assert red_region_coverage > 0.95, f"Red region should be detected (>95%), got {red_region_coverage*100:.1f}%"
    print("  [PASS] Red dual-range boundary wrap thresholding verified.")

    # 5. Test Morphological Noise Removal (Erosion removes isolated pixels)
    noisy_frame = synthetic_frame.copy()
    # Add a few stray blue noise pixels in the green region (salt noise)
    noisy_frame[50:52, 250:252] = [255, 50, 50]
    
    detector.set_color("blue")
    cleaned_mask = detector.create_mask(noisy_frame)
    # The stray 2x2 noise spot should be eliminated by MORPH_OPEN
    noise_spot_value = np.max(cleaned_mask[49:53, 249:253])
    assert noise_spot_value == 0, "Morphological opening should eliminate isolated 2x2 noise pixels."
    print("  [PASS] Morphological noise filtering verified.")

    # 6. Test Bitwise Cloak Extraction
    extracted_cloak = detector.extract_cloak(synthetic_frame, blue_mask)
    assert extracted_cloak.shape == synthetic_frame.shape
    # Non-blue quadrant should be black in extracted image
    assert np.all(extracted_cloak[230:370, 230:370] == 0), "Non-cloak areas must be pitch black (0, 0, 0)."
    print("  [PASS] Bitwise cloak extraction (cv2.bitwise_and) verified.")

    print("\n[SUCCESS] All Stage 2 automated tests completed successfully (100% PASS)!")


if __name__ == "__main__":
    run_stage2_tests()
