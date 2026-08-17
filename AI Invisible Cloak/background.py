"""
=============================================================================
Smart Invisible Cloak - Background Manager Module (Stage 1)
=============================================================================
This module manages capturing, averaging, storing, validating, and saving
the reference background image.

Viva Concepts Covered:
1. Why do we need a Background Capture?
   - The illusion of "invisibility" works by replacing the pixels of the cloak
     with the corresponding background pixels that were photographed when the
     scene was empty.
2. Why Multi-Frame Averaging (Noise Reduction)?
   - Webcams suffer from optical sensor noise and subtle exposure fluctuations.
   - Capturing 30 frames and computing the mathematical mean removes transient
     noise and produces a clean, high-fidelity reference background.
3. Shape and Data Type Compatibility:
   - The background array must match the live frame in dimensions (Height x Width x 3)
     and data type (uint8, values 0-255).
=============================================================================
"""

import os
import cv2
import numpy as np
import config


class BackgroundManager:
    """Handles static background acquisition, averaging, validation, and storage."""

    def __init__(self):
        """Initializes the background storage container."""
        # Holds the captured background as a NumPy ndarray in memory
        self._background: np.ndarray | None = None

    def capture_background(
        self, camera_handler, num_frames: int = config.BG_CAPTURE_FRAMES
    ) -> bool:
        """
        Captures a burst of video frames and computes their average to create a clean background.

        :param camera_handler: Active CameraHandler instance.
        :param num_frames: Number of consecutive frames to accumulate and average.
        :return: True if capture was successful, False otherwise.
        """
        if not camera_handler.is_running():
            print("[ERROR] Cannot capture background: Camera is not active.")
            return False

        print(f"[INFO] Capturing clean background using {num_frames} frames for noise reduction...")
        print("[INFO] Please ensure NO PERSON or dynamic object is in front of the camera!")

        frames = []
        for i in range(num_frames):
            ret, frame = camera_handler.read_frame(flip_horizontal=True)
            if ret and frame is not None:
                # Convert to float32 to prevent overflow during sum/average computation
                frames.append(frame.astype(np.float32))
            # Small delay to ensure frames are captured across distinct camera shutter cycles
            cv2.waitKey(20)

        if not frames:
            print("[ERROR] Failed to read valid frames during background capture.")
            return False

        # Compute element-wise arithmetic mean across the frame burst
        averaged_frame = np.mean(frames, axis=0)

        # Convert back to 8-bit unsigned integer (standard image pixel format 0-255)
        self._background = np.clip(averaged_frame, 0, 255).astype(np.uint8)

        height, width = self._background.shape[:2]
        print(f"[SUCCESS] Background captured successfully! Resolution: {width}x{height} pixels.")
        return True

    def set_background(self, frame: np.ndarray) -> bool:
        """
        Directly sets a single frame as the background.

        :param frame: 3D NumPy array representing an image frame.
        :return: True if valid frame was set.
        """
        if frame is None or len(frame.shape) != 3 or frame.shape[2] != 3:
            print("[ERROR] Invalid frame format provided for background.")
            return False

        self._background = frame.copy()
        return True

    def get_background(self) -> np.ndarray | None:
        """
        Returns a safe copy of the stored background frame.

        :return: A copy of the background ndarray or None if not yet captured.
        """
        if self._background is None:
            return None
        return self._background.copy()

    def has_background(self) -> bool:
        """Checks if a valid background is currently loaded in memory."""
        return self._background is not None

    def validate_with_frame(self, frame: np.ndarray) -> bool:
        """
        Checks if the stored background has the same dimensions and type as the current frame.

        :param frame: The live frame to compare against.
        :return: True if dimensions and types match, False otherwise.
        """
        if self._background is None or frame is None:
            return False
        return self._background.shape == frame.shape and self._background.dtype == frame.dtype

    def save_to_disk(self, filepath: str = config.BG_IMAGE_PATH) -> bool:
        """
        Saves the captured background image to disk as an image file (e.g. JPG or PNG).

        :param filepath: Destination file path.
        :return: True if saved successfully, False otherwise.
        """
        if not self.has_background():
            print("[ERROR] Cannot save: No background has been captured yet.")
            return False

        success = cv2.imwrite(filepath, self._background)
        if success:
            print(f"[SUCCESS] Background saved to disk at: {os.path.abspath(filepath)}")
        else:
            print(f"[ERROR] Failed to save background image to: {filepath}")
        return success

    def load_from_disk(self, filepath: str = config.BG_IMAGE_PATH) -> bool:
        """
        Loads a pre-saved background image from disk into memory.

        :param filepath: Path to the image file.
        :return: True if loaded successfully, False otherwise.
        """
        if not os.path.exists(filepath):
            print(f"[ERROR] File does not exist: {filepath}")
            return False

        img = cv2.imread(filepath)
        if img is None:
            print(f"[ERROR] Failed to decode image from: {filepath}")
            return False

        self._background = img
        height, width = self._background.shape[:2]
        print(f"[SUCCESS] Background loaded from disk: {filepath} ({width}x{height})")
        return True
