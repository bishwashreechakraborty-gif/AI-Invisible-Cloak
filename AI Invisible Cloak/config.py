"""
=============================================================================
Smart Invisible Cloak - Configuration Module (Stages 1 to 7)
=============================================================================
Centralizes all global configuration settings, hardware parameters,
HSV color presets, morphological filters, temporal smoothing, alpha
feathering, automatic ROI color-calibration, GUI / Media settings,
and optional AI Person Segmentation settings.
"""

import os
import numpy as np

# =============================================================================
# 1. Hardware / Camera Settings
# =============================================================================
CAMERA_INDEX = 0          # 0 = Default webcam. Change to 1 or 2 for external USB cameras.
FRAME_WIDTH = 640         # Horizontal resolution in pixels
FRAME_HEIGHT = 480        # Vertical resolution in pixels
FPS_TARGET = 30           # Expected camera frames per second

# =============================================================================
# 2. Background Capture Settings (Stage 1)
# =============================================================================
BG_CAPTURE_FRAMES = 30    # Number of frames to average when capturing background.
BG_IMAGE_PATH = "captured_background.jpg"  # Path to persist captured background on disk

# =============================================================================
# 3. HSV Color Segmentation Presets (Stages 2 & 5)
# =============================================================================
DEFAULT_COLOR = "blue"

# Pre-calibrated HSV Threshold Ranges:
# OpenCV HSV Scales:
#   Hue (H):        0 - 179   (Color tone on circular spectrum)
#   Saturation (S): 0 - 255   (Purity/intensity)
#   Value (V):      0 - 255   (Brightness)
HSV_COLOR_PRESETS = {
    "blue": {
        "display_name": "Blue Cloak",
        "ranges": [
            {
                "lower": np.array([90, 60, 50], dtype=np.uint8),
                "upper": np.array([130, 255, 255], dtype=np.uint8),
            }
        ],
    },
    "red": {
        "display_name": "Red Cloak",
        "ranges": [
            # Lower red near 0 degrees
            {
                "lower": np.array([0, 100, 70], dtype=np.uint8),
                "upper": np.array([10, 255, 255], dtype=np.uint8),
            },
            # Upper red near 180 degrees
            {
                "lower": np.array([170, 100, 70], dtype=np.uint8),
                "upper": np.array([180, 255, 255], dtype=np.uint8),
            },
        ],
    },
    "green": {
        "display_name": "Green Cloak",
        "ranges": [
            {
                "lower": np.array([35, 60, 50], dtype=np.uint8),
                "upper": np.array([85, 255, 255], dtype=np.uint8),
            }
        ],
    },
    "yellow": {
        "display_name": "Yellow Cloak",
        "ranges": [
            {
                "lower": np.array([20, 100, 100], dtype=np.uint8),
                "upper": np.array([35, 255, 255], dtype=np.uint8),
            }
        ],
    },
}

# =============================================================================
# 4. Mask Enhancement & Morphological Settings (Stage 4)
# =============================================================================
GAUSSIAN_BLUR_KERNEL = (7, 7)      # Pre-filter smoothing kernel for camera noise
MORPH_KERNEL_SIZE = (5, 5)         # Structuring element size for morphology
MORPH_OPEN_ITERATIONS = 2          # Opening (Erosion -> Dilation) to remove stray noise
MORPH_CLOSE_ITERATIONS = 2         # Closing (Dilation -> Erosion) to fill internal holes & shadows
MORPH_DILATE_ITERATIONS = 1        # Boundary expansion to seal edges

# Contour Filtering: Ignore small detected patches smaller than this pixel area
MIN_CONTOUR_AREA = 600             # Rejects small background objects (pens, books, toys)

# Temporal Anti-Flicker Smoothing (Exponential Moving Average)
ENABLE_TEMPORAL_SMOOTHING = True   # Default toggle state
TEMPORAL_SMOOTH_ALPHA = 0.75       # Weight of current frame mask vs previous frame (0.0 to 1.0)

# Edge Feathering / Alpha Blending (Soft Natural Transition)
ENABLE_FEATHERING = True           # Default toggle state
FEATHER_KERNEL_SIZE = (7, 7)       # Gaussian blur kernel size for alpha mask
FEATHER_SIGMA = 2.5                # Gaussian standard deviation for edge falloff

# =============================================================================
# 5. Automatic ROI Color Calibration Settings (Stage 5)
# =============================================================================
ROI_WIDTH = 140                    # Width of on-screen sampling box in pixels
ROI_HEIGHT = 140                   # Height of on-screen sampling box in pixels

# Adaptive Calibration Margins (Offsets around sampled median)
CALIB_HUE_MARGIN = 14              # +/- Hue offset (e.g. median 110 -> range [96, 124])
CALIB_SAT_MARGIN = 50              # +/- Saturation spread
CALIB_VAL_MARGIN = 60              # +/- Value/Brightness spread
CALIB_MIN_SAT = 40                 # Minimum saturation threshold to prevent white/gray false triggers
CALIB_MIN_VAL = 40                 # Minimum brightness threshold to prevent black shadow false triggers

# =============================================================================
# 6. Media Storage Settings (Stage 6)
# =============================================================================
OUTPUT_DIR = "output"
SCREENSHOTS_DIR = os.path.join(OUTPUT_DIR, "screenshots")
RECORDINGS_DIR = os.path.join(OUTPUT_DIR, "recordings")

os.makedirs(SCREENSHOTS_DIR, exist_ok=True)
os.makedirs(RECORDINGS_DIR, exist_ok=True)

RECORDING_FOURCC = "mp4v"
RECORDING_EXTENSION = ".mp4"

# =============================================================================
# 7. CustomTkinter GUI Settings (Stage 6)
# =============================================================================
APP_NAME = "Smart Invisible Cloak AI"
APP_VERSION = "v7.0 AI-Enhanced"
GUI_THEME = "dark"
GUI_COLOR_THEME = "blue"
WINDOW_WIDTH = 1260
WINDOW_HEIGHT = 820

COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_YELLOW = (0, 255, 255)
COLOR_CYAN = (255, 255, 0)
COLOR_MAGENTA = (255, 0, 255)
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)

# =============================================================================
# 8. Optional AI-Assisted Segmentation Settings (Stage 7)
# =============================================================================
# Set to False by default so OpenCV remains the primary standalone baseline
ENABLE_AI_SEGMENTATION = False
AI_CONFIDENCE_THRESHOLD = 0.5      # Minimum confidence probability (0.0 to 1.0) for person detection
AI_MODEL_SELECTION = "mediapipe"   # Selected lightweight CPU architecture
