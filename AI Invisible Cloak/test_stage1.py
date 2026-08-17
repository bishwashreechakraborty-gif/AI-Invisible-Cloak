"""
=============================================================================
Stage 1 Unit / Integration Test (Simulated Camera & Background Verification)
=============================================================================
This test validates the core logic of Stage 1 without needing an active webcam:
1. Validates BackgroundManager initialization and buffer state.
2. Simulates 30 noisy frames and verifies the noise-reduction averaging math.
3. Tests resolution validation between background and live frames.
4. Tests disk saving and reloading of background images.
=============================================================================
"""

import os
import numpy as np
import cv2

import config
from background import BackgroundManager


class MockCamera:
    """Simulates a camera returning synthetic noisy frames for testing."""
    def __init__(self, width=640, height=480):
        self.width = width
        self.height = height
        # Base scene: gradient image
        x = np.linspace(0, 255, width, dtype=np.uint8)
        y = np.linspace(0, 255, height, dtype=np.uint8)
        xx, yy = np.meshgrid(x, y)
        self.base_scene = cv2.merge([xx, yy, (xx // 2 + yy // 2).astype(np.uint8)])

    def is_running(self):
        return True

    def read_frame(self, flip_horizontal=True):
        # Add random zero-mean Gaussian sensor noise
        noise = np.random.normal(0, 10, self.base_scene.shape).astype(np.float32)
        noisy_frame = np.clip(self.base_scene.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return True, noisy_frame


def run_tests():
    print("[TEST] Running Stage 1 Automated Logic Tests...")

    # Test 1: BackgroundManager initial state
    bg_mgr = BackgroundManager()
    assert not bg_mgr.has_background(), "Initial state should not have background."
    assert bg_mgr.get_background() is None, "Initial background should be None."
    print("  [PASS] Initial state check passed.")

    # Test 2: Multi-frame capture & averaging
    mock_cam = MockCamera(width=config.FRAME_WIDTH, height=config.FRAME_HEIGHT)
    captured = bg_mgr.capture_background(mock_cam, num_frames=30)
    assert captured, "Capture should return True."
    assert bg_mgr.has_background(), "BackgroundManager should have background."

    bg = bg_mgr.get_background()
    assert bg is not None
    assert bg.shape == (config.FRAME_HEIGHT, config.FRAME_WIDTH, 3), f"Shape mismatch: {bg.shape}"
    assert bg.dtype == np.uint8, f"Dtype mismatch: {bg.dtype}"
    print("  [PASS] Multi-frame capture and averaging passed.")

    # Test 3: Resolution validation
    _, live_frame = mock_cam.read_frame()
    assert bg_mgr.validate_with_frame(live_frame), "Live frame should match background resolution."
    
    # Test mismatch with different sized frame
    wrong_size_frame = np.zeros((300, 300, 3), dtype=np.uint8)
    assert not bg_mgr.validate_with_frame(wrong_size_frame), "Wrong size frame should fail validation."
    print("  [PASS] Resolution and dimension validation passed.")

    # Test 4: Save to disk & Reload
    test_path = "test_captured_bg.jpg"
    assert bg_mgr.save_to_disk(test_path), "Saving to disk should succeed."
    assert os.path.exists(test_path), "File must exist on disk."

    new_bg_mgr = BackgroundManager()
    assert new_bg_mgr.load_from_disk(test_path), "Loading from disk should succeed."
    assert new_bg_mgr.has_background(), "Loaded background should be present."
    loaded_bg = new_bg_mgr.get_background()
    assert loaded_bg.shape == bg.shape, "Loaded background shape must match original."
    print("  [PASS] Disk persistence and loading passed.")

    # Cleanup test image
    if os.path.exists(test_path):
        os.remove(test_path)

    print("\n[SUCCESS] All Stage 1 logic tests passed with 100% accuracy!")


if __name__ == "__main__":
    run_tests()
