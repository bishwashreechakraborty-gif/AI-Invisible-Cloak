"""
=============================================================================
Smart Invisible Cloak - AI Segmenter Module (Stage 7)
=============================================================================
This module provides an optional, lightweight AI-assisted person segmentation
subsystem using MediaPipe Selfie Segmentation. It combines the neural-network
person mask (P) with the classical HSV cloak mask (C) using logical intersection:

                   M_AI = P ∩ C = cv2.bitwise_and(P, C)

Viva Concepts Covered in this Module:
-----------------------------------------------------------------------------
1. Why Add AI Person Segmentation to Classical HSV Masking?
   - Classical HSV segmentation has a fundamental physical limitation:
     "Color Ambiguity". If the room contains blue walls, blue curtains, a blue
     book, or blue jeans, the HSV threshold marks all of them as cloak.
   - The AI segmenter generates a spatial prior (Person Mask, P) that identifies
     the exact boundary of the human body.
   - Taking the logical AND (P ∩ C) guarantees that ONLY color pixels lying ON
     THE PERSON'S BODY are converted into the invisible cloak, completely
     eliminating background false positives.

2. Model Selection: MediaPipe vs YOLOv8-seg vs DeepLab:
   - MediaPipe Selfie Segmentation is chosen because it requires < 5 MB of memory,
     runs in ~10-15 ms on standard laptop CPUs via XNNPACK/TFLite, and maintains
     30-60+ FPS without needing a heavy discrete NVIDIA CUDA GPU or multi-gigabyte
     PyTorch dependencies.

3. Graceful Fallback:
   - If AI segmentation is disabled or the model fails to load, the system
     seamlessly falls back to the original OpenCV HSV-only pipeline without
     crashing or interruption.
=============================================================================
"""

import cv2
import numpy as np
import config

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False


class AISegmenter:
    """Provides lightweight neural-network person segmentation and hybrid mask fusion."""

    def __init__(self, confidence_threshold: float = config.AI_CONFIDENCE_THRESHOLD):
        """
        Initialize the AI segmenter.

        :param confidence_threshold: Float (0.0 to 1.0) probability threshold for person class.
        """
        self.confidence_threshold = confidence_threshold
        self.model = None
        self._is_initialized = False
        self._load_error_message = ""

        # Try to load the model on initialization
        self.load_model()

    def load_model(self) -> bool:
        """
        Initializes the MediaPipe Selfie Segmentation model.

        :return: True if loaded successfully, False otherwise.
        """
        if not MEDIAPIPE_AVAILABLE:
            self._load_error_message = "MediaPipe package not found."
            self._is_initialized = False
            return False

        try:
            # Model selection: 0 = General (faster, 256x256), 1 = Landscape (higher precision)
            if hasattr(mp, "solutions") and hasattr(mp.solutions, "selfie_segmentation"):
                self.model = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=0)
                self._is_initialized = True
                print("[INFO] MediaPipe Selfie Segmentation AI Model loaded successfully.")
                return True
            else:
                self._load_error_message = "MediaPipe solutions API not accessible."
                self._is_initialized = False
                return False

        except Exception as e:
            self._load_error_message = str(e)
            self._is_initialized = False
            print(f"[WARNING] Could not initialize AI model: {e}")
            return False

    def is_available(self) -> bool:
        """Returns True if the AI model is loaded and ready for inference."""
        return self._is_initialized and self.model is not None

    def get_model_name(self) -> str:
        """Returns descriptive name of the neural network model."""
        return "MediaPipe Selfie Segmenter (CPU Optimized)"

    def generate_person_mask(self, frame: np.ndarray) -> np.ndarray:
        """
        Runs neural-network inference on the frame and returns a binary person mask.

        :param frame: Live BGR video frame from webcam.
        :return: 2D binary mask (uint8) where Person = 255 and Background = 0.
        """
        if frame is None:
            return np.zeros((config.FRAME_HEIGHT, config.FRAME_WIDTH), dtype=np.uint8)

        if not self.is_available():
            # Fallback: Return all 255s (entire frame allowed, identical to OpenCV baseline)
            return np.full(frame.shape[:2], 255, dtype=np.uint8)

        try:
            # MediaPipe expects RGB image
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False  # Performance optimization

            results = self.model.process(rgb_frame)

            if results.segmentation_mask is None:
                return np.full(frame.shape[:2], 255, dtype=np.uint8)

            # Convert continuous confidence probabilities (0.0 to 1.0) into binary mask (0 or 255)
            prob_mask = results.segmentation_mask
            person_mask = (prob_mask >= self.confidence_threshold).astype(np.uint8) * 255

            return person_mask

        except Exception as e:
            print(f"[WARNING] AI inference failed on frame: {e}")
            return np.full(frame.shape[:2], 255, dtype=np.uint8)

    def combine_with_cloak_mask(
        self,
        person_mask: np.ndarray,
        cloak_mask: np.ndarray,
    ) -> np.ndarray:
        """
        Computes the spatial logical intersection between the AI Person Mask (P)
        and the HSV Cloak Mask (C): M_AI = P ∩ C.

        :param person_mask: Binary mask of human body (255 for person, 0 for background).
        :param cloak_mask: Binary mask of target cloak color (255 for color, 0 for non-color).
        :return: Hybrid binary mask where only cloak fabric ON THE PERSON is 255.
        """
        if person_mask is None or cloak_mask is None:
            return cloak_mask if cloak_mask is not None else np.zeros((config.FRAME_HEIGHT, config.FRAME_WIDTH), dtype=np.uint8)

        # Spatial intersection (bitwise AND)
        # Keeps pixels that are BOTH (A) On the person AND (B) Matching the cloak color
        hybrid_mask = cv2.bitwise_and(person_mask, cloak_mask)
        return hybrid_mask
