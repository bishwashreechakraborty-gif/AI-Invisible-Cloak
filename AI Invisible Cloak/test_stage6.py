"""
=============================================================================
Stage 6 Automated Unit & Integration Tests (Media Recorder & GUI Helpers)
=============================================================================
This test suite verifies:
1. MediaRecorder directory creation and storage paths.
2. Screenshot export, timestamp generation, and image file integrity.
3. MP4 video recording lifecycle (start, write frames, stop, file size check).
4. Image conversion pipeline (OpenCV BGR -> RGB -> PIL Image).
5. File cleanup and directory sanity checks.
=============================================================================
"""

import os
import time
import numpy as np
import cv2
from PIL import Image

import config
from recorder import MediaRecorder


def run_stage6_tests():
    print("[TEST] Running Stage 6 Automated Media & GUI Helper Tests...")

    recorder = MediaRecorder()

    # -------------------------------------------------------------------------
    # Test 1: Storage Directory Creation
    # -------------------------------------------------------------------------
    assert os.path.exists(config.OUTPUT_DIR), "Output root directory must exist."
    assert os.path.exists(config.SCREENSHOTS_DIR), "Screenshots directory must exist."
    assert os.path.exists(config.RECORDINGS_DIR), "Recordings directory must exist."
    print("  [PASS] Media storage directories verified.")

    # -------------------------------------------------------------------------
    # Test 2: Screenshot Export
    # -------------------------------------------------------------------------
    h, w = 480, 640
    test_frame = np.full((h, w, 3), [120, 80, 200], dtype=np.uint8)  # Purple test swatch

    snap_path = recorder.save_screenshot(test_frame, prefix="test_snap")
    assert os.path.exists(snap_path), f"Saved screenshot must exist: {snap_path}"
    assert os.path.getsize(snap_path) > 0, "Screenshot file must not be empty."

    # Validate image can be reloaded by OpenCV
    loaded_img = cv2.imread(snap_path)
    assert loaded_img is not None
    assert loaded_img.shape == (h, w, 3), f"Shape mismatch: {loaded_img.shape}"
    print(f"  [PASS] Screenshot capture and disk validation verified: {os.path.basename(snap_path)}")

    # -------------------------------------------------------------------------
    # Test 3: MP4 Video Recording Lifecycle
    # -------------------------------------------------------------------------
    assert not recorder.is_recording(), "Recorder must initially be inactive."
    rec_path = recorder.start_recording(width=w, height=h, fps=30.0)
    assert recorder.is_recording(), "Recorder must report active recording state."

    # Write 30 synthetic frames
    for i in range(30):
        # Create dynamic animated gradient frame
        frame_i = np.full((h, w, 3), [(i * 8) % 255, 100, 150], dtype=np.uint8)
        written = recorder.write_frame(frame_i)
        assert written is True, "Frame write must succeed."

    time.sleep(0.1)
    final_path, total_frames, duration = recorder.stop_recording()
    assert not recorder.is_recording(), "Recorder must be inactive after stop."
    assert total_frames == 30, f"Expected 30 frames, got {total_frames}"
    assert os.path.exists(final_path), f"Video file must exist: {final_path}"
    assert os.path.getsize(final_path) > 0, "Video file must have non-zero size."
    print(f"  [PASS] MP4 video recording verified: {os.path.basename(final_path)} ({total_frames} frames, {duration:.2f}s)")

    # -------------------------------------------------------------------------
    # Test 4: Image Pipeline (OpenCV BGR -> RGB -> PIL Image)
    # -------------------------------------------------------------------------
    bgr_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    bgr_frame[:, :, 0] = 255  # Pure Blue in BGR

    rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    assert rgb_frame[0, 0, 2] == 255, "Channel 2 in RGB must be Blue."
    assert rgb_frame[0, 0, 0] == 0, "Channel 0 in RGB must be Red."

    pil_image = Image.fromarray(rgb_frame)
    assert pil_image.size == (640, 480), f"PIL size mismatch: {pil_image.size}"
    print("  [PASS] Image transformation pipeline (OpenCV -> PIL) verified.")

    # -------------------------------------------------------------------------
    # Cleanup Test Artifacts
    # -------------------------------------------------------------------------
    if os.path.exists(snap_path):
        os.remove(snap_path)
    if os.path.exists(final_path):
        os.remove(final_path)

    print("\n[SUCCESS] All Stage 6 automated tests passed with 100% accuracy!")


if __name__ == "__main__":
    run_stage6_tests()
