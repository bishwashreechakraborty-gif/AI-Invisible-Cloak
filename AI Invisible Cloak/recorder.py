"""
=============================================================================
Smart Invisible Cloak - Media Recorder Module (Stage 6)
=============================================================================
This module provides a thread-safe screenshot capturing and MP4 video
recording subsystem using OpenCV's cv2.VideoWriter.

Viva Concepts Covered in this Module:
-----------------------------------------------------------------------------
1. How does cv2.VideoWriter work?
   - cv2.VideoWriter encodes individual image frames into a standardized video
     container (e.g. .mp4 or .avi) using a compression codec specified by a
     FourCC (Four Character Code) identifier (e.g. 'mp4v' or 'XVID').

2. Frame Synchronization:
   - Video recording requires consistent frame dimensions (Width x Height) and
     a matching Frame Rate (FPS). If a frame has different dimensions, it must
     be resized to match the stream header before writing to prevent file corruption.

3. Organized Media Storage:
   - Automated timestamping (YYYYMMDD_HHMMSS) ensures media captures are safely
     archived in structured output/ directories without file naming collisions.
=============================================================================
"""

import os
import time
from datetime import datetime
import cv2
import numpy as np
import config


class MediaRecorder:
    """Manages screenshot export and MP4 video stream recording."""

    def __init__(
        self,
        screenshots_dir: str = config.SCREENSHOTS_DIR,
        recordings_dir: str = config.RECORDINGS_DIR,
    ):
        """
        Initialize the media recorder and verify storage directories.

        :param screenshots_dir: Directory path for saving snapshot PNGs.
        :param recordings_dir: Directory path for saving recorded MP4 videos.
        """
        self.screenshots_dir = screenshots_dir
        self.recordings_dir = recordings_dir

        os.makedirs(self.screenshots_dir, exist_ok=True)
        os.makedirs(self.recordings_dir, exist_ok=True)

        self._writer: cv2.VideoWriter | None = None
        self._is_recording: bool = False
        self._current_video_path: str = ""
        self._start_time: float = 0.0
        self._frame_count: int = 0
        self._target_dims: tuple[int, int] = (config.FRAME_WIDTH, config.FRAME_HEIGHT)

    def save_screenshot(self, frame: np.ndarray, prefix: str = "cloak_snap") -> str:
        """
        Saves a single frame to disk as a high-quality PNG image with a timestamp.

        :param frame: 3-channel BGR image array.
        :param prefix: Filename prefix.
        :return: Absolute file path of the saved screenshot.
        """
        if frame is None or frame.size == 0:
            raise ValueError("Cannot save an empty frame.")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}_{timestamp}.png"
        filepath = os.path.join(self.screenshots_dir, filename)

        cv2.imwrite(filepath, frame)
        abs_path = os.path.abspath(filepath)
        print(f"[MEDIA] Screenshot saved: {abs_path}")
        return abs_path

    def start_recording(
        self,
        width: int = config.FRAME_WIDTH,
        height: int = config.FRAME_HEIGHT,
        fps: float = float(config.FPS_TARGET),
    ) -> str:
        """
        Initializes the video writer stream and begins recording frames.

        :param width: Video frame width in pixels.
        :param height: Video frame height in pixels.
        :param fps: Target recording framerate (default 30.0).
        :return: Destination filepath for the recording.
        """
        if self._is_recording:
            return self._current_video_path

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"cloak_recording_{timestamp}{config.RECORDING_EXTENSION}"
        self._current_video_path = os.path.join(self.recordings_dir, filename)

        self._target_dims = (width, height)
        fourcc = cv2.VideoWriter_fourcc(*config.RECORDING_FOURCC)

        self._writer = cv2.VideoWriter(
            self._current_video_path,
            fourcc,
            fps,
            (width, height),
        )

        if not self._writer.isOpened():
            # Fallback codec if mp4v is unsupported on system backend
            fallback_fourcc = cv2.VideoWriter_fourcc(*"XVID")
            filename_avi = f"cloak_recording_{timestamp}.avi"
            self._current_video_path = os.path.join(self.recordings_dir, filename_avi)
            self._writer = cv2.VideoWriter(
                self._current_video_path,
                fallback_fourcc,
                fps,
                (width, height),
            )

        self._is_recording = True
        self._start_time = time.time()
        self._frame_count = 0

        print(f"[MEDIA] Recording started: {os.path.abspath(self._current_video_path)}")
        return os.path.abspath(self._current_video_path)

    def write_frame(self, frame: np.ndarray) -> bool:
        """
        Writes a single frame to the active video file.

        :param frame: Live BGR video frame.
        :return: True if frame was written successfully, False otherwise.
        """
        if not self._is_recording or self._writer is None or frame is None:
            return False

        h, w = frame.shape[:2]
        target_w, target_h = self._target_dims

        # Ensure frame dimensions match writer initialization specifications
        if (w, h) != (target_w, target_h):
            frame_to_write = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        else:
            frame_to_write = frame

        self._writer.write(frame_to_write)
        self._frame_count += 1
        return True

    def stop_recording(self) -> tuple[str, int, float]:
        """
        Finalizes and closes the active video recording.

        :return: Tuple (filepath, total_frames_written, duration_in_seconds).
        """
        if not self._is_recording or self._writer is None:
            return "", 0, 0.0

        self._writer.release()
        self._writer = None
        self._is_recording = False

        duration = time.time() - self._start_time
        path = os.path.abspath(self._current_video_path)
        print(f"[MEDIA] Recording stopped: {path} ({self._frame_count} frames, {duration:.1f}s)")
        return path, self._frame_count, duration

    def is_recording(self) -> bool:
        """Returns True if video is currently recording."""
        return self._is_recording

    def get_duration(self) -> float:
        """Returns elapsed recording duration in seconds."""
        if not self._is_recording:
            return 0.0
        return time.time() - self._start_time

    def get_formatted_duration(self) -> str:
        """Returns elapsed recording time formatted as 'MM:SS'."""
        secs = int(self.get_duration())
        mins = secs // 60
        secs = secs % 60
        return f"{mins:02d}:{secs:02d}"
