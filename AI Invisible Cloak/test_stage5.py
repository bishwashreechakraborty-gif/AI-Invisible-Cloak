"""
=============================================================================
Stage 5 Automated Unit & Integration Tests (ROI Calibration & Multi-Color)
=============================================================================
This test suite verifies:
1. ColorCalibrator ROI coordinate calculation across different frame dimensions.
2. Automatic calibration of a synthetic Blue swatch.
3. Automatic calibration of a synthetic Red swatch (boundary wrap-around logic).
4. Custom preset registration and switching in ColorDetector.
5. Direct index selection (Presets 1 to 5).
6. End-to-end integration with InvisibilityEngine.
=============================================================================
"""

import numpy as np
import cv2

import config
from calibrator import ColorCalibrator
from color_detector import ColorDetector
from invisibility_engine import InvisibilityEngine


def run_stage5_tests():
    print("[TEST] Running Stage 5 Automated Calibration & Multi-Color Tests...")

    calibrator = ColorCalibrator(roi_width=140, roi_height=140)
    detector = ColorDetector(default_color="blue")
    engine = InvisibilityEngine()

    # -------------------------------------------------------------------------
    # Test 1: ROI Coordinate Calculation
    # -------------------------------------------------------------------------
    h, w = 480, 640
    x1, y1, x2, y2 = calibrator.get_roi_bounds((h, w))
    assert x1 == (640 // 2 - 70) and x2 == (640 // 2 + 70)
    assert y1 == (480 // 2 - 70) and y2 == (480 // 2 + 70)
    assert (x2 - x1) == 140 and (y2 - y1) == 140
    print("  [PASS] ROI coordinate calculation verified.")

    # -------------------------------------------------------------------------
    # Test 2: Auto-Calibration on Synthetic Blue Fabric
    # -------------------------------------------------------------------------
    test_frame = np.full((h, w, 3), [100, 100, 100], dtype=np.uint8)
    # Fill ROI with a rich Royal Blue fabric swatch (BGR = [220, 80, 20])
    test_frame[y1:y2, x1:x2] = [220, 80, 20]

    blue_preset = calibrator.calibrate(test_frame)
    assert blue_preset is not None
    assert "ranges" in blue_preset
    telemetry = blue_preset["telemetry"]
    # In OpenCV HSV, pure blue is Hue ~ 110-125
    assert 100 <= telemetry["h_median"] <= 130, f"Expected Blue hue, got {telemetry['h_median']}"
    assert not telemetry["has_wrap"], "Blue should not wrap around 0/180."
    print("  [PASS] Auto-calibration of synthetic blue fabric verified.")

    # -------------------------------------------------------------------------
    # Test 3: Auto-Calibration on Synthetic Red Fabric (Wrap-Around Detection)
    # -------------------------------------------------------------------------
    red_frame = np.full((h, w, 3), [100, 100, 100], dtype=np.uint8)
    # Fill ROI with a Crimson Red fabric swatch (BGR = [20, 20, 230] -> Hue ~ 0-5)
    red_frame[y1:y2, x1:x2] = [20, 20, 230]

    red_preset = calibrator.calibrate(red_frame)
    assert red_preset is not None
    red_telemetry = red_preset["telemetry"]
    assert red_telemetry["has_wrap"] is True, "Red hue near 0 must trigger boundary wrap-around."
    assert len(red_preset["ranges"]) == 2, "Red preset must generate dual threshold ranges."
    print("  [PASS] Auto-calibration of Red fabric and dual-range wrap-around verified.")

    # -------------------------------------------------------------------------
    # Test 4: Custom Preset Registration & Switching in ColorDetector
    # -------------------------------------------------------------------------
    success = detector.register_custom_preset(blue_preset)
    assert success is True
    assert detector.get_active_color_key() == "custom"
    assert detector.has_custom_calib is True

    # Test direct preset selection by index (1: Blue, 2: Red, 3: Green, 4: Yellow, 5: Custom)
    assert detector.select_color_by_index(1) is True
    assert detector.get_active_color_key() == "blue"

    assert detector.select_color_by_index(2) is True
    assert detector.get_active_color_key() == "red"

    assert detector.select_color_by_index(3) is True
    assert detector.get_active_color_key() == "green"

    assert detector.select_color_by_index(4) is True
    assert detector.get_active_color_key() == "yellow"

    assert detector.select_color_by_index(5) is True
    assert detector.get_active_color_key() == "custom"
    print("  [PASS] Custom preset registration and direct 1-5 selection verified.")

    # -------------------------------------------------------------------------
    # Test 5: End-to-End Pipeline Integration with Invisibility Engine
    # -------------------------------------------------------------------------
    # Frame with custom calibrated blue square
    custom_cloak_frame = np.full((h, w, 3), [150, 150, 150], dtype=np.uint8)
    custom_cloak_frame[150:350, 200:400] = [220, 80, 20]  # Calibrated blue

    detector.select_color_by_index(5)  # Custom preset
    mask = detector.create_mask(custom_cloak_frame)
    assert np.mean(mask[180:320, 230:370]) / 255.0 > 0.90, "Custom calibrated mask must detect swatch (>90%)."

    synth_bg = np.full((h, w, 3), [0, 255, 0], dtype=np.uint8)
    output, _, _ = engine.blend(custom_cloak_frame, synth_bg, mask, use_feathering=True)
    # Output center should be green (replaced by background)
    assert output[250, 300, 1] > 200, "Invisibility output must replace custom swatch with background."
    print("  [PASS] End-to-end auto-calibration and invisibility blending verified.")

    print("\n[SUCCESS] All Stage 5 automated tests passed with 100% accuracy!")


if __name__ == "__main__":
    run_stage5_tests()
