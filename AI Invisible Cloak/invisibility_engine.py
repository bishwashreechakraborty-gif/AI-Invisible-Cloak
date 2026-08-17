"""
=============================================================================
Smart Invisible Cloak - Invisibility Engine Module (Stage 4)
=============================================================================
This module performs spatial background replacement with support for:
1. Crisp Bitwise Blending (Stage 3 logic)
2. Soft Alpha Feathering / Edge Smoothing (Stage 4 enhancement)

Viva Concepts Covered in this Module:
-----------------------------------------------------------------------------
1. Why does Crisp Binary Masking cause "Scissor-Cut" Artifacts?
   - In pure binary masking, every pixel is strictly 0 (100% person) or 255
     (100% background).
   - Real-world camera pixels along the edges of the cloak are a mixture of both
     due to optical blur, sub-pixel fabric motion, and lens physics.
   - Forcing a sharp step function results in jagged, pixelated "cutout" borders.

2. Edge Feathering / Alpha Blending:
   - We apply a Gaussian Blur to the binary mask to transform sharp step transitions
     into a smooth continuous gradient α(x, y) ∈ [0.0, 1.0].
   - Final composite equation:
     Final(x, y) = α(x, y) * Background(x, y) + (1.0 - α(x, y)) * LiveFrame(x, y)
   - Result: Seamless optical transition where the background effortlessly melts into
     the person's body without harsh boundary lines.
=============================================================================
"""

import cv2
import numpy as np
import config


class InvisibilityEngine:
    """Combines live video, cloak mask, and background using bitwise blending or alpha feathering."""

    def __init__(self):
        """Initializes the invisibility blending engine with feathering parameters."""
        self.enable_feathering: bool = config.ENABLE_FEATHERING
        self.feather_kernel: tuple[int, int] = config.FEATHER_KERNEL_SIZE
        self.feather_sigma: float = config.FEATHER_SIGMA

    def toggle_feathering(self) -> bool:
        """Toggles Edge Feathering (Alpha Blending) ON or OFF."""
        self.enable_feathering = not self.enable_feathering
        status = "ENABLED" if self.enable_feathering else "DISABLED"
        print(f"[ACTION] Edge Feathering / Alpha Blending: {status}")
        return self.enable_feathering

    def generate_inverse_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Computes the bitwise inverse of the binary cloak mask.

        :param mask: Binary mask (255 for cloak, 0 for background).
        :return: Inverted mask (0 for cloak, 255 for background and subject).
        """
        if mask is None:
            return np.array([], dtype=np.uint8)
        return cv2.bitwise_not(mask)

    def generate_alpha_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Converts a binary 0/255 mask into a smoothed 3-channel floating-point alpha map [0.0, 1.0].

        :param mask: Binary mask (uint8).
        :return: 3-channel float32 alpha mask of shape (Height, Width, 3).
        """
        # Apply Gaussian Blur to smooth the boundary transition
        blurred = cv2.GaussianBlur(mask, self.feather_kernel, self.feather_sigma)
        # Normalize to [0.0, 1.0] range
        alpha_1ch = blurred.astype(np.float32) / 255.0
        # Stack into 3 channels for element-wise broadcasting across B, G, R
        return cv2.merge([alpha_1ch, alpha_1ch, alpha_1ch])

    def blend(
        self,
        current_frame: np.ndarray,
        background: np.ndarray | None,
        mask: np.ndarray,
        use_feathering: bool | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Executes the invisibility blending pipeline.

        :param current_frame: Live BGR video frame.
        :param background: Pre-captured clean background.
        :param mask: Binary cloak mask.
        :param use_feathering: Override toggle for alpha feathering (defaults to self.enable_feathering).
        :return: Tuple (final_output, bg_segment, fg_segment).
        """
        if current_frame is None:
            raise ValueError("current_frame cannot be None.")

        # Fallback: If no background has been captured yet, return the live frame
        if background is None:
            blank_bg = np.zeros_like(current_frame)
            return current_frame.copy(), blank_bg, current_frame.copy()

        # Dimension Safety: Ensure background matches live frame resolution
        if background.shape[:2] != current_frame.shape[:2]:
            h, w = current_frame.shape[:2]
            background = cv2.resize(background, (w, h), interpolation=cv2.INTER_LINEAR)

        feather = self.enable_feathering if use_feathering is None else use_feathering

        if feather:
            # ---------------------------------------------------------------------
            # Mode A: Soft Alpha Feathering / Edge Blending (Stage 4)
            # ---------------------------------------------------------------------
            alpha = self.generate_alpha_mask(mask)
            bg_float = background.astype(np.float32)
            fg_float = current_frame.astype(np.float32)

            # Linear interpolation: Output = alpha * BG + (1 - alpha) * FG
            final_float = alpha * bg_float + (1.0 - alpha) * fg_float
            final_output = np.clip(final_float, 0, 255).astype(np.uint8)

            bg_segment = np.clip(alpha * bg_float, 0, 255).astype(np.uint8)
            fg_segment = np.clip((1.0 - alpha) * fg_float, 0, 255).astype(np.uint8)

            return final_output, bg_segment, fg_segment

        else:
            # ---------------------------------------------------------------------
            # Mode B: Crisp Bitwise Blending (Stage 3)
            # ---------------------------------------------------------------------
            mask_inv = self.generate_inverse_mask(mask)
            res1 = cv2.bitwise_and(background, background, mask=mask)
            res2 = cv2.bitwise_and(current_frame, current_frame, mask=mask_inv)
            final_output = cv2.add(res1, res2)

            return final_output, res1, res2
