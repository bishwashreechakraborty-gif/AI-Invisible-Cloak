"""
=============================================================================
Smart Invisible Cloak - Camera Handler Module (Stage 1)
=============================================================================
This module encapsulates all webcam interactions using OpenCV (cv2).
It manages device initialization, frame acquisition, camera warm-up,
mirroring (horizontal flip), and safe hardware release.

Viva Concepts Covered:
1. cv2.VideoCapture(index):
   - Opens the video capturing device using hardware drivers (DirectShow on Windows).
   - Index 0 is typically the primary built-in webcam.
2. Frame Representation:
   - OpenCV frames are 3D NumPy arrays with shape: (Height, Width, Channels).
   - OpenCV uses BGR (Blue, Green, Red) channel ordering by default instead of RGB.
3. Warm-up Period:
   - Digital webcams need a few initial frames to adjust auto-exposure and white balance.
4. Horizontal Flipping (Mirror Effect):
   - cv2.flip(frame, 1) flips around the y-axis so moving left moves left on screen.
=============================================================================
"""

import time
import cv2
import numpy as np
import config


class CameraHandler:
    """Manages webcam initialization, frame capturing, and resource cleanup."""

    def __init__(
        self,
        camera_index: int = config.CAMERA_INDEX,
        width: int = config.FRAME_WIDTH,
        height: int = config.FRAME_HEIGHT,
    ):
        """
        Initialize camera parameters without opening the device immediately.

        :param camera_index: Integer index of the webcam (0 for default).
        :param width: Target frame width in pixels.
        :param height: Target frame height in pixels.
        """
        self.camera_index = camera_index
        self.target_width = width
        self.target_height = height
        self.cap: cv2.VideoCapture | None = None
        self.actual_width: int = 0
        self.actual_height: int = 0

    def start(self) -> bool:
        """
        Opens the webcam stream, configures resolution, and performs camera warm-up.

        :return: True if the camera opened and initialized successfully, False otherwise.
        """
        print(f"[INFO] Initializing webcam at index {self.camera_index}...")

        # Open camera stream (On Windows, cv2.CAP_DSHOW can also be used if default backend has lag)
        self.cap = cv2.VideoCapture(self.camera_index)

        # Check if the webcam device opened properly
        if not self.cap.isOpened():
            print(f"[ERROR] Could not open webcam at index {self.camera_index}.")
            print("[HINT] Ensure no other application (Zoom, Teams, Camera app) is using the webcam.")
            return False

        # Request resolution configuration from driver
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_height)

        # Warm up the camera sensor (discard first few frames while auto-exposure settles)
        print("[INFO] Warming up camera sensor...")
        time.sleep(1.0)
        for _ in range(10):
            ret, _ = self.cap.read()
            if not ret:
                print("[WARNING] Warm-up frame read failed.")
                break

        # Read one actual frame to verify dimensions reported by hardware
        ret, test_frame = self.cap.read()
        if not ret or test_frame is None:
            print("[ERROR] Camera opened but failed to deliver valid video frames.")
            self.release()
            return False

        self.actual_height, self.actual_width = test_frame.shape[:2]
        print(f"[SUCCESS] Webcam active. Frame Resolution: {self.actual_width}x{self.actual_height}")
        return True

    def read_frame(self, flip_horizontal: bool = True) -> tuple[bool, np.ndarray | None]:
        """
        Grabs, decodes, and returns the next video frame.

        :param flip_horizontal: If True, flips frame horizontally (mirror effect for user feed).
        :return: Tuple (ret, frame) where ret is a boolean indicating success and frame is a NumPy ndarray.
        """
        if self.cap is None or not self.cap.isOpened():
            return False, None

        # ret: boolean (True if frame is successfully read)
        # frame: NumPy ndarray with shape (Height, Width, 3) in BGR format
        ret, frame = self.cap.read()

        if not ret or frame is None:
            return False, None

        # Flip horizontally (around vertical y-axis) so it acts like a mirror
        if flip_horizontal:
            frame = cv2.flip(frame, 1)

        return True, frame

    def get_resolution(self) -> tuple[int, int]:
        """Returns the actual resolution as a tuple of (width, height)."""
        return self.actual_width, self.actual_height

    def is_running(self) -> bool:
        """Returns whether the video capture object is open and active."""
        return self.cap is not None and self.cap.isOpened()

    def release(self):
        """Releases the camera device and frees system resources."""
        if self.cap is not None:
            if self.cap.isOpened():
                self.cap.release()
            self.cap = None
            print("[INFO] Webcam hardware successfully released.")

    def __enter__(self):
        """Allows usage as a Python context manager (with CameraHandler() as cam:)."""
        if not self.start():
            raise RuntimeError("Failed to start camera inside context manager.")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Ensures camera cleanup on context exit."""
        self.release()
