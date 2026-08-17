"""
=============================================================================
Smart Invisible Cloak - Color Detector Module (Stages 2, 4 & 5)
=============================================================================
This module handles HSV color space conversion, thresholding, multi-color
preset switching, dynamic custom preset registration, and advanced classical
computer-vision mask refinement.
"""

import cv2
import numpy as np
import config


class ColorDetector:
    """Detects and refines cloak masks using HSV thresholding, morphology, contour filtering, and EMA."""

    def __init__(self, default_color: str = config.DEFAULT_COLOR):
        """
        Initialize the color detector with configuration parameters and presets.

        :param default_color: Initial active preset key (e.g. 'blue', 'red', 'green', 'yellow').
        """
        # Shallow copy presets so custom presets can be added without mutating config global
        self.presets = dict(config.HSV_COLOR_PRESETS)
        self.active_color_key: str = default_color if default_color in self.presets else "blue"
        self.has_custom_calib: bool = False

        # Morphological structuring element
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, config.MORPH_KERNEL_SIZE)

        # Contour filtering threshold
        self.min_contour_area: int = config.MIN_CONTOUR_AREA

        # Temporal smoothing buffer
        self.prev_mask_float: np.ndarray | None = None
        self.enable_temporal_smoothing: bool = config.ENABLE_TEMPORAL_SMOOTHING
        self.temporal_alpha: float = config.TEMPORAL_SMOOTH_ALPHA

    def reset_temporal_buffer(self):
        """Clears the temporal smoothing memory buffer."""
        self.prev_mask_float = None

    def register_custom_preset(self, preset_dict: dict) -> bool:
        """
        Registers or updates the dynamic 'custom' color preset generated from auto-calibration.

        :param preset_dict: Calibrated dictionary containing 'display_name' and 'ranges'.
        :return: True if registered and activated successfully.
        """
        if not preset_dict or "ranges" not in preset_dict:
            return False

        self.presets["custom"] = preset_dict
        self.active_color_key = "custom"
        self.has_custom_calib = True
        self.reset_temporal_buffer()
        print(f"[ACTION] Activated custom calibrated color preset: {preset_dict.get('display_name')}")
        return True

    def set_color(self, color_name: str) -> bool:
        """
        Switches the active color detection preset by name/key.

        :param color_name: Name of color to activate ('blue', 'red', 'green', 'yellow', 'custom').
        :return: True if switched successfully, False if color not found.
        """
        color_lower = color_name.lower().strip()
        if color_lower in self.presets:
            self.active_color_key = color_lower
            self.reset_temporal_buffer()
            print(f"[INFO] Active detection color set to: {self.get_display_name()}")
            return True
        else:
            print(f"[WARNING] Unknown color '{color_name}'. Available: {list(self.presets.keys())}")
            return False

    def select_color_by_index(self, index: int) -> bool:
        """
        Selects color preset directly by 1-based or 0-based integer index (1: Blue, 2: Red, 3: Green, 4: Yellow, 5: Custom).

        :param index: 1-based index (1 to N).
        :return: True if valid index selected.
        """
        color_keys = list(self.presets.keys())
        idx_zero = index - 1
        if 0 <= idx_zero < len(color_keys):
            return self.set_color(color_keys[idx_zero])
        return False

    def cycle_next_color(self) -> str:
        """Cycles to the next available color preset in sequence and returns its key."""
        color_keys = list(self.presets.keys())
        current_index = color_keys.index(self.active_color_key)
        next_index = (current_index + 1) % len(color_keys)
        self.active_color_key = color_keys[next_index]
        self.reset_temporal_buffer()
        print(f"[ACTION] Switched color to: {self.get_display_name()}")
        return self.active_color_key

    def toggle_temporal_smoothing(self) -> bool:
        """Toggles temporal anti-flicker smoothing ON or OFF."""
        self.enable_temporal_smoothing = not self.enable_temporal_smoothing
        self.reset_temporal_buffer()
        status = "ENABLED" if self.enable_temporal_smoothing else "DISABLED"
        print(f"[ACTION] Temporal Anti-Flicker Smoothing: {status}")
        return self.enable_temporal_smoothing

    def get_active_color_key(self) -> str:
        """Returns the current active color key (e.g., 'blue', 'custom')."""
        return self.active_color_key

    def get_display_name(self) -> str:
        """Returns the user-friendly display name of the active color."""
        return self.presets[self.active_color_key]["display_name"]

    def get_available_colors(self) -> list[str]:
        """Returns a list of all registered color preset keys."""
        return list(self.presets.keys())

    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Applies gentle Gaussian Blur to reduce high-frequency camera noise and grain
        before color space conversion.

        :param frame: Input BGR image frame.
        :return: Blurred BGR frame.
        """
        return cv2.GaussianBlur(frame, config.GAUSSIAN_BLUR_KERNEL, 0)

    def create_mask(
        self,
        frame: np.ndarray,
        return_raw: bool = False,
    ) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
        """
        Processes the frame through the complete Stage 4 refinement pipeline:
        HSV Thresholding -> Morphological Opening -> Morphological Closing ->
        Contour Area Filtering -> Boundary Dilation -> Temporal Anti-Flicker Smoothing.

        :param frame: Live BGR video frame from webcam.
        :param return_raw: If True, returns a tuple (refined_mask, raw_mask).
        :return: Refined binary mask (or tuple if return_raw is True).
        """
        if frame is None:
            empty = np.zeros((config.FRAME_HEIGHT, config.FRAME_WIDTH), dtype=np.uint8)
            return (empty, empty) if return_raw else empty

        # Step 1: Gaussian blur & BGR to HSV
        blurred = self.preprocess_frame(frame)
        hsv_frame = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        # Step 2: Color Thresholding across ranges
        active_preset = self.presets[self.active_color_key]
        raw_mask = None

        for r in active_preset["ranges"]:
            lower = r["lower"]
            upper = r["upper"]
            mask_segment = cv2.inRange(hsv_frame, lower, upper)
            if raw_mask is None:
                raw_mask = mask_segment
            else:
                raw_mask = cv2.bitwise_or(raw_mask, mask_segment)

        # Step 3: Morphological Opening (remove speckles outside cloak)
        opened_mask = cv2.morphologyEx(
            raw_mask,
            cv2.MORPH_OPEN,
            self.kernel,
            iterations=config.MORPH_OPEN_ITERATIONS,
        )

        # Step 4: Morphological Closing (fill internal holes & shadows)
        closed_mask = cv2.morphologyEx(
            opened_mask,
            cv2.MORPH_CLOSE,
            self.kernel,
            iterations=config.MORPH_CLOSE_ITERATIONS,
        )

        # Step 5: Contour Extraction & Minimum-Area Filtering
        contours, _ = cv2.findContours(
            closed_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        contour_filtered_mask = np.zeros_like(closed_mask)
        valid_contours = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area >= self.min_contour_area:
                valid_contours.append(cnt)

        if valid_contours:
            cv2.drawContours(
                contour_filtered_mask,
                valid_contours,
                -1,
                255,
                thickness=cv2.FILLED,
            )
        else:
            contour_filtered_mask = closed_mask

        # Step 6: Gentle Boundary Dilation
        dilated_mask = cv2.dilate(
            contour_filtered_mask,
            self.kernel,
            iterations=config.MORPH_DILATE_ITERATIONS,
        )

        # Step 7: Temporal Anti-Flicker Smoothing (EMA)
        current_float = dilated_mask.astype(np.float32)

        if self.enable_temporal_smoothing and self.prev_mask_float is not None:
            if self.prev_mask_float.shape == current_float.shape:
                smoothed_float = (
                    self.temporal_alpha * current_float
                    + (1.0 - self.temporal_alpha) * self.prev_mask_float
                )
                self.prev_mask_float = smoothed_float
                final_mask = (smoothed_float > 127.5).astype(np.uint8) * 255
            else:
                self.prev_mask_float = current_float
                final_mask = dilated_mask
        else:
            self.prev_mask_float = current_float
            final_mask = dilated_mask

        if return_raw:
            return final_mask, raw_mask
        return final_mask

    def extract_cloak(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Segments and isolates the detected cloak fabric by applying bitwise AND with the binary mask.

        :param frame: Live BGR image.
        :param mask: Binary mask (255 for cloak, 0 for other).
        :return: BGR image showing only the cloak region.
        """
        if frame is None or mask is None:
            return np.zeros_like(frame)
        return cv2.bitwise_and(frame, frame, mask=mask)
