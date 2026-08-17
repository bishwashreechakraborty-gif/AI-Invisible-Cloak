"""
=============================================================================
Smart Invisible Cloak - Application Entry Point (Stage 6)
=============================================================================
Launches the Stage 6 CustomTkinter Desktop Application, bringing together
all 6 modular layers of the computer-vision and media architecture:
1. CameraHandler        (Webcam frame ingestion)
2. BackgroundManager    (Multi-frame background noise reduction)
3. ColorDetector        (Multi-preset HSV segmentation & morphology)
4. ColorCalibrator      (Interactive ROI statistical auto-calibration)
5. InvisibilityEngine   (Gaussian Alpha Feathering & bitwise blending)
6. MediaRecorder        (Timestamped screenshots & MP4 video encoding)
7. SmartInvisibleCloakApp (Asynchronous CustomTkinter Desktop GUI)
=============================================================================
"""

import sys
from gui_app import SmartInvisibleCloakApp


def main():
    """Entry point for the Smart Invisible Cloak desktop application."""
    print("=" * 75)
    print("  SMART INVISIBLE CLOAK - STAGE 6 (PROFESSIONAL DESKTOP GUI)")
    print("=" * 75)
    print("Launching CustomTkinter Desktop Application...")
    
    app = SmartInvisibleCloakApp()
    app.mainloop()


if __name__ == "__main__":
    main()