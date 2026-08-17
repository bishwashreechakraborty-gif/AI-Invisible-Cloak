"""
=============================================================================
Smart Invisible Cloak - Color Calibrator Module (Stage 5)
=============================================================================
This module provides an automatic, interactive Region-of-Interest (ROI) color
calibration system. It allows users to sample real-world fabric colors under
any ambient lighting condition and automatically derives optimal HSV bounds.

Viva Concepts Covered in this Module:
-----------------------------------------------------------------------------
1. What is a Region of Interest (ROI)?
   - An ROI is a specific rectangular spatial sub-array within the larger frame
     matrix: Frame[y1:y2, x1:x2].
   - Rather than processing the whole image, we isolate the fabric swatch in
     the center of the camera view to avoid background interference.

2. Statistical Threshold Estimation:
   - Consumer fabrics have micro-texture variations and lighting gradients.
   - We calculate the Central Tendency (Median) for Hue (H), Saturation (S),
     and Value (V) to resist outlier noise.
   - Adaptive tolerance margins (±ΔH, ±ΔS, ±ΔV) are applied around the median.

3. Hue Wrap-Around Handling:
   - In OpenCV, Hue is mapped cyclically from 0 to 179 (360° / 2).
   - If a sampled red/magenta fabric has a median near 0 (e.g. H=5), subtracting
     ΔH=14 would yield a negative value (-9).
   - The calibrator automatically detects this and splits the threshold into two
     valid disjoint sub-ranges: [0, 19] and [171, 180].

4. Why Ambient Lighting Dictates HSV Thresholds:
   - Fluorescent lighting (cool, high blue temperature), tungsten lamps (warm,
     high yellow), and sunlight all alter the measured Hue and Saturation of the
     exact same physical cloth. Auto-calibration solves this in real time.
=============================================================================
"""

import cv2
import numpy as np
import config


class ColorCalibrator:
    """Handles on-screen ROI rendering and statistical HSV color calibration."""

    def __init__(
        self,
        roi_width: int = config.ROI_WIDTH,
        roi_height: int = config.ROI_HEIGHT,
        hue_margin: int = config.CALIB_HUE_MARGIN,
        sat_margin: int = config.CALIB_SAT_MARGIN,
        val_margin: int = config.CALIB_VAL_MARGIN,
    ):
        """
        Initialize calibration parameters.

        :param roi_width: Width of the central sampling box in pixels.
        :param roi_height: Height of the central sampling box in pixels.
        :param hue_margin: Hue boundary margin offset (±ΔH).
        :param sat_margin: Saturation boundary margin offset (±ΔS).
        :param val_margin: Value boundary margin offset (±ΔV).
        """
        self.roi_width = roi_width
        self.roi_height = roi_height
        self.hue_margin = hue_margin
        self.sat_margin = sat_margin
        self.val_margin = val_margin

        # Telemetry of latest calibration
        self.latest_telemetry: dict | None = None

    def get_roi_bounds(self, frame_shape: tuple[int, int]) -> tuple[int, int, int, int]:
        """
        Computes the pixel coordinates (x1, y1, x2, y2) for the centered ROI box.

        :param frame_shape: Shape of the frame (Height, Width).
        :return: (x1, y1, x2, y2) bounding box.
        """
        h, w = frame_shape[:2]
        cx, cy = w // 2, h // 2
        x1 = max(0, cx - self.roi_width // 2)
        y1 = max(0, cy - self.roi_height // 2)
        x2 = min(w, cx + self.roi_width // 2)
        y2 = min(h, cy + self.roi_height // 2)
        return x1, y1, x2, y2

    def draw_roi_overlay(self, frame: np.ndarray, is_calibrated: bool = False) -> np.ndarray:
        """
        Renders an on-screen targeting reticle with corner brackets and helper text.

        :param frame: Image frame to draw on.
        :param is_calibrated: Boolean indicating if a custom calibration has been performed.
        :return: Annotated frame with ROI guide.
        """
        annotated = frame.copy()
        x1, y1, x2, y2 = self.get_roi_bounds(annotated.shape)

        box_color = config.COLOR_GREEN if is_calibrated else config.COLOR_YELLOW
        thickness = 2
        corner_len = 18

        # Draw Corner Brackets (High-Tech HUD Reticle)
        # Top-Left
        cv2.line(annotated, (x1, y1), (x1 + corner_len, y1), box_color, thickness)
        cv2.line(annotated, (x1, y1), (x1, y1 + corner_len), box_color, thickness)
        # Top-Right
        cv2.line(annotated, (x2, y1), (x2 - corner_len, y1), box_color, thickness)
        cv2.line(annotated, (x2, y1), (x2, y1 + corner_len), box_color, thickness)
        # Bottom-Left
        cv2.line(annotated, (x1, y2), (x1 + corner_len, y2), box_color, thickness)
        cv2.line(annotated, (x1, y2), (x1, y2 - corner_len), box_color, thickness)
        # Bottom-Right
        cv2.line(annotated, (x2, y2), (x2 - corner_len, y2), box_color, thickness)
        cv2.line(annotated, (x2, y2), (x2, y2 - corner_len), box_color, thickness)

        # Center crosshair dot
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.circle(annotated, (cx, cy), 3, box_color, -1)

        # Helper text below ROI
        prompt_text = "[ROI] Place Fabric Inside Box & Press 'K'"
        cv2.putText(
            annotated,
            prompt_text,
            (x1 - 50, y2 + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            box_color,
            1,
            cv2.LINE_AA,
        )

        return annotated

    def calibrate(self, frame: np.ndarray) -> dict | None:
        """
        Samples the ROI pixels, calculates statistical HSV bounds, and generates a new preset.

        :param frame: Live BGR video frame containing fabric inside the ROI.
        :return: Preset dictionary formatted for HSV_COLOR_PRESETS, or None if calibration fails.
        """
        if frame is None:
            return None

        x1, y1, x2, y2 = self.get_roi_bounds(frame.shape)
        # Crop the spatial Region of Interest (ROI)
        roi_bgr = frame[y1:y2, x1:x2]

        if roi_bgr.size == 0:
            return None

        # Pre-filter ROI with Gaussian blur to smooth yarn texture
        roi_blurred = cv2.GaussianBlur(roi_bgr, (5, 5), 0)
        roi_hsv = cv2.cvtColor(roi_blurred, cv2.COLOR_BGR2HSV)

        # Flatten channel arrays
        h_vals = roi_hsv[:, :, 0].flatten()
        s_vals = roi_hsv[:, :, 1].flatten()
        v_vals = roi_hsv[:, :, 2].flatten()

        # Discard extreme shadow / washed out pixels to prevent biased medians
        valid_mask = (v_vals >= 25) & (s_vals >= 25)
        if np.sum(valid_mask) < 100:
            # If fabric is extremely dark/black or washed out white
            valid_mask = np.ones_like(v_vals, dtype=bool)

        h_valid = h_vals[valid_mask]
        s_valid = s_vals[valid_mask]
        v_valid = v_vals[valid_mask]

        # Compute medians for central tendency
        h_med = int(np.median(h_valid))
        s_med = int(np.median(s_valid))
        v_med = int(np.median(v_valid))

        # Compute Saturation and Value bounds
        s_min = max(config.CALIB_MIN_SAT, s_med - self.sat_margin)
        s_max = min(255, s_med + self.sat_margin)

        v_min = max(config.CALIB_MIN_VAL, v_med - self.val_margin)
        v_max = min(255, v_med + self.val_margin)

        # Compute Hue bounds with Wrap-Around logic
        ranges = []
        h_lower_calc = h_med - self.hue_margin
        h_upper_calc = h_med + self.hue_margin

        if h_lower_calc < 0:
            # Case A: Red wrap-around near 0 (e.g. H_med = 5, lower = -9)
            # Range 1: [0, h_upper_calc]
            ranges.append({
                "lower": np.array([0, s_min, v_min], dtype=np.uint8),
                "upper": np.array([min(179, h_upper_calc), s_max, v_max], dtype=np.uint8),
            })
            # Range 2: [180 + h_lower_calc, 180]
            ranges.append({
                "lower": np.array([180 + h_lower_calc, s_min, v_min], dtype=np.uint8),
                "upper": np.array([180, s_max, v_max], dtype=np.uint8),
            })
            wrap_text = " (Red-Wrap Active)"

        elif h_upper_calc > 179:
            # Case B: Red wrap-around near 180 (e.g. H_med = 175, upper = 189)
            ranges.append({
                "lower": np.array([max(0, h_lower_calc), s_min, v_min], dtype=np.uint8),
                "upper": np.array([179, s_max, v_max], dtype=np.uint8),
            })
            ranges.append({
                "lower": np.array([0, s_min, v_min], dtype=np.uint8),
                "upper": np.array([h_upper_calc - 180, s_max, v_max], dtype=np.uint8),
            })
            wrap_text = " (Red-Wrap Active)"

        else:
            # Case C: Standard single range
            ranges.append({
                "lower": np.array([h_lower_calc, s_min, v_min], dtype=np.uint8),
                "upper": np.array([h_upper_calc, s_max, v_max], dtype=np.uint8),
            })
            wrap_text = ""

        telemetry = {
            "h_median": h_med,
            "s_median": s_med,
            "v_median": v_med,
            "h_range": (max(0, h_lower_calc), min(179, h_upper_calc)),
            "s_range": (s_min, s_max),
            "v_range": (v_min, v_max),
            "has_wrap": len(ranges) > 1,
        }
        self.latest_telemetry = telemetry

        preset = {
            "display_name": f"Custom Calibrated (H:{h_med}){wrap_text}",
            "ranges": ranges,
            "telemetry": telemetry,
        }

        print(f"[SUCCESS] Auto-Calibration Complete!")
        print(f"  - Sampled Medians: Hue={h_med}, Saturation={s_med}, Value={v_med}")
        print(f"  - Thresholds: H=[{h_lower_calc}-{h_upper_calc}], S=[{s_min}-{s_max}], V=[{v_min}-{v_max}]")

        return preset
