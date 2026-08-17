"""
=============================================================================
Smart Invisible Cloak - Desktop GUI Application (Stages 1 to 7)
=============================================================================
A polished, modern desktop application built with CustomTkinter.
Integrates all 7 computer-vision and AI modules:
1. CameraHandler        (Webcam frame ingestion)
2. BackgroundManager    (Multi-frame background noise reduction)
3. ColorDetector        (Multi-preset HSV segmentation & morphology)
4. ColorCalibrator      (Interactive ROI statistical auto-calibration)
5. InvisibilityEngine   (Gaussian Alpha Feathering & bitwise blending)
6. MediaRecorder        (Timestamped screenshots & MP4 video encoding)
7. AISegmenter          (Optional MediaPipe AI Person Segmentation & Hybrid Fusion)
8. SmartInvisibleCloakApp (Asynchronous CustomTkinter Desktop GUI)
=============================================================================
"""

import os
import subprocess
import platform
import time
import cv2
import numpy as np
from PIL import Image
import customtkinter as ctk

import config
from camera import CameraHandler
from background import BackgroundManager
from color_detector import ColorDetector
from invisibility_engine import InvisibilityEngine
from calibrator import ColorCalibrator
from recorder import MediaRecorder
from ai_segmenter import AISegmenter

# Set global appearance mode and color theme
ctk.set_appearance_mode(config.GUI_THEME)
ctk.set_default_color_theme(config.GUI_COLOR_THEME)


class SmartInvisibleCloakApp(ctk.CTk):
    """Main Desktop Application Window for the Smart Invisible Cloak."""

    def __init__(self):
        """Initializes GUI widgets, CV/AI components, and the video processing loop."""
        super().__init__()

        # ---------------------------------------------------------------------
        # 1. Window Configuration
        # ---------------------------------------------------------------------
        self.title(f"{config.APP_NAME} - {config.APP_VERSION}")
        self.geometry(f"{config.WINDOW_WIDTH}x{config.WINDOW_HEIGHT}")
        self.minsize(1050, 720)

        # ---------------------------------------------------------------------
        # 2. Initialize Computer Vision & AI Components (Stages 1-7)
        # ---------------------------------------------------------------------
        self.camera = CameraHandler(
            camera_index=config.CAMERA_INDEX,
            width=config.FRAME_WIDTH,
            height=config.FRAME_HEIGHT,
        )
        self.bg_manager = BackgroundManager()
        self.detector = ColorDetector(default_color=config.DEFAULT_COLOR)
        self.engine = InvisibilityEngine()
        self.calibrator = ColorCalibrator()
        self.recorder = MediaRecorder()
        self.ai_segmenter = AISegmenter()

        # ---------------------------------------------------------------------
        # 3. Application State Variables
        # ---------------------------------------------------------------------
        self.is_camera_running = False
        self.invisibility_enabled = True
        self.ai_enabled = config.ENABLE_AI_SEGMENTATION  # Default False (OpenCV baseline)
        self.active_view_mode = "Invisible Cloak"
        self.status_message = "Ready. Click 'Start Camera' to begin."

        # FPS calculation variables
        self.prev_frame_time = time.time()
        self.current_fps = 0.0

        # Frame buffers for debug visualizations
        self._last_rendered_frame = None

        # ---------------------------------------------------------------------
        # 4. Build GUI Layout
        # ---------------------------------------------------------------------
        self._build_ui_layout()

        # Handle window close event
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        # Start non-blocking video scheduling loop
        self.after(50, self._video_loop)

    def _build_ui_layout(self):
        """Constructs the sidebar controls and main video viewport."""
        self.grid_columnconfigure(0, weight=0, minsize=350)  # Left Sidebar
        self.grid_columnconfigure(1, weight=1)               # Right Viewport
        self.grid_rowconfigure(0, weight=1)

        # =====================================================================
        # Left Sidebar (Controls Panel)
        # =====================================================================
        self.sidebar = ctk.CTkScrollableFrame(self, corner_radius=12, width=340)
        self.sidebar.grid(row=0, column=0, padx=(16, 8), pady=16, sticky="nsew")

        # --- Header ---
        self.lbl_title = ctk.CTkLabel(
            self.sidebar,
            text="✨ INVISIBLE CLOAK AI",
            font=ctk.CTkFont(size=20, weight="bold"),
        )
        self.lbl_title.pack(padx=10, pady=(10, 2), anchor="w")

        self.lbl_sub = ctk.CTkLabel(
            self.sidebar,
            text=f"Computer Vision & AI System • {config.APP_VERSION}",
            font=ctk.CTkFont(size=12),
            text_color="gray",
        )
        self.lbl_sub.pack(padx=10, pady=(0, 12), anchor="w")

        # --- Card 1: Camera Hardware Controls ---
        self.card_cam = ctk.CTkFrame(self.sidebar, corner_radius=10)
        self.card_cam.pack(fill="x", padx=6, pady=5)

        ctk.CTkLabel(self.card_cam, text="📷 Camera Stream", font=ctk.CTkFont(size=13, weight="bold")).pack(padx=12, pady=(8, 3), anchor="w")
        self.btn_camera = ctk.CTkButton(
            self.card_cam,
            text="Start Camera Feed",
            fg_color="#2b8a3e",
            hover_color="#237032",
            font=ctk.CTkFont(weight="bold"),
            command=self._toggle_camera,
        )
        self.btn_camera.pack(fill="x", padx=12, pady=(3, 8))

        # --- Card 2: Background Acquisition (Stage 1) ---
        self.card_bg = ctk.CTkFrame(self.sidebar, corner_radius=10)
        self.card_bg.pack(fill="x", padx=6, pady=5)

        ctk.CTkLabel(self.card_bg, text="🖼️ Background Reference", font=ctk.CTkFont(size=13, weight="bold")).pack(padx=12, pady=(8, 2), anchor="w")
        self.lbl_bg_status = ctk.CTkLabel(self.card_bg, text="Status: Not Captured", text_color="#fcc419", font=ctk.CTkFont(size=11))
        self.lbl_bg_status.pack(padx=12, pady=1, anchor="w")

        self.btn_bg_capture = ctk.CTkButton(
            self.card_bg,
            text="Capture Background (30 Frames)",
            command=self._action_capture_bg,
        )
        self.btn_bg_capture.pack(fill="x", padx=12, pady=(3, 8))

        # --- Card 3: Color & ROI Auto-Calibration (Stages 2 & 5) ---
        self.card_color = ctk.CTkFrame(self.sidebar, corner_radius=10)
        self.card_color.pack(fill="x", padx=6, pady=5)

        ctk.CTkLabel(self.card_color, text="🎨 Cloak Color & Calibration", font=ctk.CTkFont(size=13, weight="bold")).pack(padx=12, pady=(8, 2), anchor="w")
        
        ctk.CTkLabel(self.card_color, text="Select Predefined Preset:", font=ctk.CTkFont(size=11), text_color="gray").pack(padx=12, pady=(2, 1), anchor="w")
        self.opt_color = ctk.CTkOptionMenu(
            self.card_color,
            values=["Blue Cloak", "Red Cloak", "Green Cloak", "Yellow Cloak", "Custom Calibrated"],
            command=self._on_color_selected,
        )
        self.opt_color.set("Blue Cloak")
        self.opt_color.pack(fill="x", padx=12, pady=(0, 6))

        self.btn_calibrate = ctk.CTkButton(
            self.card_color,
            text="🎯 Auto-Calibrate from ROI",
            fg_color="#1864ab",
            hover_color="#14528d",
            command=self._action_calibrate_roi,
        )
        self.btn_calibrate.pack(fill="x", padx=12, pady=(2, 4))

        self.lbl_calib_info = ctk.CTkLabel(
            self.card_color,
            text="Place cloth in center box & click Calibrate.",
            font=ctk.CTkFont(size=10),
            text_color="gray",
            wraplength=280,
        )
        self.lbl_calib_info.pack(padx=12, pady=(0, 8), anchor="w")

        # --- Card 4: AI & Pipeline Enhancements (Stages 3, 4, 7) ---
        self.card_fx = ctk.CTkFrame(self.sidebar, corner_radius=10)
        self.card_fx.pack(fill="x", padx=6, pady=5)

        ctk.CTkLabel(self.card_fx, text="⚡ Pipeline Enhancements", font=ctk.CTkFont(size=13, weight="bold")).pack(padx=12, pady=(8, 4), anchor="w")

        self.sw_invisibility = ctk.CTkSwitch(
            self.card_fx,
            text="Enable Invisibility Effect",
            command=self._on_toggle_invisibility,
        )
        self.sw_invisibility.select()
        self.sw_invisibility.pack(padx=12, pady=3, anchor="w")

        # AI Segmentation Switch (Stage 7 Feature)
        self.sw_ai = ctk.CTkSwitch(
            self.card_fx,
            text="🤖 AI Person Masking (Hybrid)",
            command=self._on_toggle_ai,
            progress_color="#862e9c",
        )
        if self.ai_enabled:
            self.sw_ai.select()
        self.sw_ai.pack(padx=12, pady=3, anchor="w")

        self.sw_feathering = ctk.CTkSwitch(
            self.card_fx,
            text="Soft Edge Feathering (Alpha)",
            command=self._on_toggle_feathering,
        )
        self.sw_feathering.select()
        self.sw_feathering.pack(padx=12, pady=3, anchor="w")

        self.sw_temporal = ctk.CTkSwitch(
            self.card_fx,
            text="Anti-Flicker Smoothing (EMA)",
            command=self._on_toggle_temporal,
        )
        self.sw_temporal.select()
        self.sw_temporal.pack(padx=12, pady=(3, 8), anchor="w")

        # --- Card 5: Viewport Display Mode ---
        self.card_view = ctk.CTkFrame(self.sidebar, corner_radius=10)
        self.card_view.pack(fill="x", padx=6, pady=5)

        ctk.CTkLabel(self.card_view, text="🖥️ Viewport Layout", font=ctk.CTkFont(size=13, weight="bold")).pack(padx=12, pady=(8, 4), anchor="w")
        self.seg_view = ctk.CTkSegmentedButton(
            self.card_view,
            values=["Invisible Cloak", "2x2 Grid", "AI Debug View", "Before/After"],
            command=self._on_view_mode_changed,
        )
        self.seg_view.set("Invisible Cloak")
        self.seg_view.pack(fill="x", padx=8, pady=(2, 8))

        # --- Card 6: Media Capture & Recording (Stage 6) ---
        self.card_media = ctk.CTkFrame(self.sidebar, corner_radius=10)
        self.card_media.pack(fill="x", padx=6, pady=5)

        ctk.CTkLabel(self.card_media, text="📸 Media & Recording", font=ctk.CTkFont(size=13, weight="bold")).pack(padx=12, pady=(8, 4), anchor="w")

        self.btn_screenshot = ctk.CTkButton(
            self.card_media,
            text="📸 Save Screenshot",
            fg_color="#495057",
            hover_color="#343a40",
            command=self._action_take_screenshot,
        )
        self.btn_screenshot.pack(fill="x", padx=12, pady=3)

        self.btn_record = ctk.CTkButton(
            self.card_media,
            text="🔴 Start MP4 Recording",
            fg_color="#c92a2a",
            hover_color="#a61e1e",
            command=self._action_toggle_recording,
        )
        self.btn_record.pack(fill="x", padx=12, pady=(3, 8))

        # --- Footer Actions ---
        self.btn_open_folder = ctk.CTkButton(
            self.sidebar,
            text="📁 Open Output Folder",
            fg_color="#343a40",
            hover_color="#212529",
            command=self._action_open_folder,
        )
        self.btn_open_folder.pack(fill="x", padx=6, pady=(8, 3))

        self.btn_exit = ctk.CTkButton(
            self.sidebar,
            text="Exit Application",
            fg_color="#862e9c",
            hover_color="#672477",
            command=self._on_close,
        )
        self.btn_exit.pack(fill="x", padx=6, pady=(3, 12))

        # =====================================================================
        # Right Main Panel (Video Viewport & Telemetry)
        # =====================================================================
        self.main_panel = ctk.CTkFrame(self, corner_radius=12)
        self.main_panel.grid(row=0, column=1, padx=(8, 16), pady=16, sticky="nsew")
        self.main_panel.grid_rowconfigure(1, weight=1)
        self.main_panel.grid_columnconfigure(0, weight=1)

        # --- Telemetry Header ---
        self.telemetry_bar = ctk.CTkFrame(self.main_panel, height=44, corner_radius=8)
        self.telemetry_bar.grid(row=0, column=0, padx=12, pady=(12, 6), sticky="ew")
        self.telemetry_bar.grid_columnconfigure(0, weight=1)
        self.telemetry_bar.grid_columnconfigure(1, weight=0)

        self.lbl_header_info = ctk.CTkLabel(
            self.telemetry_bar,
            text="🔴 Camera: Offline | Mode: OpenCV Baseline",
            font=ctk.CTkFont(size=12, weight="bold"),
            anchor="w",
        )
        self.lbl_header_info.grid(row=0, column=0, padx=12, pady=8, sticky="w")

        self.lbl_fps_badge = ctk.CTkLabel(
            self.telemetry_bar,
            text="FPS: 0.0",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#69db7c",
        )
        self.lbl_fps_badge.grid(row=0, column=1, padx=14, pady=8, sticky="e")

        # --- Video Viewport Canvas ---
        self.viewport_container = ctk.CTkFrame(self.main_panel, corner_radius=10, fg_color="#141517")
        self.viewport_container.grid(row=1, column=0, padx=12, pady=6, sticky="nsew")
        self.viewport_container.grid_rowconfigure(0, weight=1)
        self.viewport_container.grid_columnconfigure(0, weight=1)

        self.lbl_video = ctk.CTkLabel(
            self.viewport_container,
            text="Webcam Feed Inactive\n\nClick 'Start Camera Feed' on the left to begin.",
            font=ctk.CTkFont(size=16),
            text_color="gray",
        )
        self.lbl_video.grid(row=0, column=0, sticky="nsew")

        # --- Bottom Status / Notification Toast Bar ---
        self.status_bar = ctk.CTkFrame(self.main_panel, height=36, corner_radius=8)
        self.status_bar.grid(row=2, column=0, padx=12, pady=(6, 12), sticky="ew")

        self.lbl_status = ctk.CTkLabel(
            self.status_bar,
            text=self.status_message,
            font=ctk.CTkFont(size=12),
            anchor="w",
        )
        self.lbl_status.pack(fill="x", padx=12, pady=6)

    # =========================================================================
    # Video Processing & Non-blocking Frame Loop
    # =========================================================================
    def _video_loop(self):
        """Non-blocking video acquisition, hybrid AI/CV pipeline, and rendering loop."""
        try:
            if self.is_camera_running and self.camera.is_running():
                # 1. Read live frame from webcam
                success, raw_frame = self.camera.read_frame(flip_horizontal=True)

                if success and raw_frame is not None:
                    # 2. Calculate instantaneous FPS
                    curr_time = time.time()
                    dt = curr_time - self.prev_frame_time
                    if dt > 0:
                        self.current_fps = 0.9 * self.current_fps + 0.1 * (1.0 / dt) if self.current_fps > 0 else 1.0 / dt
                    self.prev_frame_time = curr_time

                    # 3. Stage 2 & 4: Classical HSV Cloak Mask (C)
                    refined_mask, raw_mask = self.detector.create_mask(raw_frame, return_raw=True)

                    # 4. Stage 7: Optional AI Person Mask (P) & Hybrid Intersection (P ∩ C)
                    if self.ai_enabled and self.ai_segmenter.is_available():
                        person_mask = self.ai_segmenter.generate_person_mask(raw_frame)
                        effective_mask = self.ai_segmenter.combine_with_cloak_mask(person_mask, refined_mask)
                        ai_tag = "🤖 Mode: AI Hybrid (P ∩ C)"
                    else:
                        person_mask = np.full_like(refined_mask, 255)
                        effective_mask = refined_mask
                        ai_tag = "⚡ Mode: OpenCV Baseline"

                    # 5. Stage 3 & 4: Invisibility Blending
                    stored_bg = self.bg_manager.get_background()
                    if self.invisibility_enabled:
                        invisible_frame, _, _ = self.engine.blend(
                            current_frame=raw_frame,
                            background=stored_bg,
                            mask=effective_mask,
                        )
                    else:
                        invisible_frame = raw_frame.copy()

                    # 6. Write to active video recording if enabled
                    if self.recorder.is_recording():
                        self.recorder.write_frame(invisible_frame)
                        self.btn_record.configure(
                            text=f"⏹️ Stop Recording ({self.recorder.get_formatted_duration()})"
                        )

                    # 7. Compose Viewport Frame based on Active Layout
                    composed_frame = self._compose_viewport_frame(
                        raw_frame=raw_frame,
                        raw_mask=raw_mask,
                        refined_mask=refined_mask,
                        person_mask=person_mask,
                        hybrid_mask=effective_mask,
                        invisible_frame=invisible_frame,
                        has_bg=self.bg_manager.has_background(),
                    )

                    # Save current frame reference for screenshots
                    self._last_rendered_frame = composed_frame.copy()

                    # 8. Convert OpenCV BGR Image -> RGB -> PIL Image -> CTkImage
                    self._render_frame_on_viewport(composed_frame)

                    # 9. Update Telemetry Bar
                    rec_tag = f" | 🔴 REC {self.recorder.get_formatted_duration()}" if self.recorder.is_recording() else ""
                    self.lbl_header_info.configure(
                        text=f"🟢 Live ({config.FRAME_WIDTH}x{config.FRAME_HEIGHT}) | {ai_tag} | Preset: {self.detector.get_display_name()}{rec_tag}"
                    )
                    self.lbl_fps_badge.configure(text=f"FPS: {self.current_fps:.1f}")

        except Exception as e:
            print(f"[ERROR] Error inside video loop: {e}")

        # Schedule next frame in 10 milliseconds
        self.after(10, self._video_loop)

    def _compose_viewport_frame(
        self,
        raw_frame: np.ndarray,
        raw_mask: np.ndarray,
        refined_mask: np.ndarray,
        person_mask: np.ndarray,
        hybrid_mask: np.ndarray,
        invisible_frame: np.ndarray,
        has_bg: bool,
    ) -> np.ndarray:
        """Composes the multi-panel layout or single viewport based on current user selection."""
        mode = self.active_view_mode

        if mode == "Invisible Cloak":
            if not has_bg and self.invisibility_enabled:
                return self._draw_warning_box(invisible_frame, "BACKGROUND NOT CAPTURED", "Click 'Capture Background' on the left")
            return invisible_frame

        elif mode == "2x2 Grid":
            # 2x2 Multi-View Layout
            cell_size = (380, 280)
            feed_roi = self.calibrator.draw_roi_overlay(raw_frame, is_calibrated=self.detector.has_custom_calib)

            p1 = cv2.resize(feed_roi, cell_size)
            p2 = cv2.resize(cv2.cvtColor(raw_mask, cv2.COLOR_GRAY2BGR), cell_size)
            p3 = cv2.resize(cv2.cvtColor(hybrid_mask, cv2.COLOR_GRAY2BGR), cell_size)
            p4 = cv2.resize(invisible_frame, cell_size)

            p1 = self._annotate_card(p1, "1. Live Feed (ROI Guide)")
            p2 = self._annotate_card(p2, "2. Raw HSV Mask (Before)")
            p3 = self._annotate_card(p3, f"3. Hybrid Mask ({self.detector.get_display_name()})")
            p4 = self._annotate_card(p4, "4. Final Invisible Cloak")

            top = np.hstack([p1, p2])
            bottom = np.hstack([p3, p4])
            return np.vstack([top, bottom])

        elif mode == "AI Debug View":
            # Stage 7 AI Debug Comparative View (4-Panel Pipeline Comparison)
            cell_size = (380, 280)
            p1 = cv2.resize(raw_frame, cell_size)
            p2 = cv2.resize(cv2.cvtColor(refined_mask, cv2.COLOR_GRAY2BGR), cell_size)
            p3 = cv2.resize(cv2.cvtColor(person_mask, cv2.COLOR_GRAY2BGR), cell_size)
            p4 = cv2.resize(cv2.cvtColor(hybrid_mask, cv2.COLOR_GRAY2BGR), cell_size)

            p1 = self._annotate_card(p1, "1. Live Input Frame")
            p2 = self._annotate_card(p2, "2. OpenCV HSV Cloak Mask (C)")
            p3 = self._annotate_card(p3, "3. AI Person Mask (P)")
            p4 = self._annotate_card(p4, "4. Combined Hybrid Mask (P ∩ C)")

            top = np.hstack([p1, p2])
            bottom = np.hstack([p3, p4])
            return np.vstack([top, bottom])

        elif mode == "Before/After":
            # Side-by-side Before and After Mask Comparison
            preview_size = (480, 360)
            m_raw = cv2.resize(cv2.cvtColor(raw_mask, cv2.COLOR_GRAY2BGR), preview_size)
            m_ref = cv2.resize(cv2.cvtColor(hybrid_mask, cv2.COLOR_GRAY2BGR), preview_size)

            m_raw = self._annotate_card(m_raw, "BEFORE: Raw HSV Mask (Noisy)")
            m_ref = self._annotate_card(m_ref, f"AFTER: Refined Hybrid Mask ({self.detector.get_display_name()})")
            return np.hstack([m_raw, m_ref])

        else:  # Fallback to Live Feed
            return self.calibrator.draw_roi_overlay(raw_frame, is_calibrated=self.detector.has_custom_calib)

    def _render_frame_on_viewport(self, frame_bgr: np.ndarray):
        """Converts an OpenCV BGR frame into a CTkImage and updates the UI canvas label."""
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)

        container_w = max(400, self.viewport_container.winfo_width() - 20)
        container_h = max(300, self.viewport_container.winfo_height() - 20)

        img_w, img_h = pil_img.size
        scale = min(container_w / img_w, container_h / img_h)
        disp_w = int(img_w * scale)
        disp_h = int(img_h * scale)

        ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(disp_w, disp_h))
        self.lbl_video.configure(image=ctk_img, text="")
        self.lbl_video.image = ctk_img

    def _annotate_card(self, image: np.ndarray, title: str) -> np.ndarray:
        """Helper to draw header bars on multi-panel preview cards."""
        card = image.copy()
        w = card.shape[1]
        cv2.rectangle(card, (0, 0), (w, 30), (0, 0, 0), -1)
        cv2.putText(card, title, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.rectangle(card, (0, 0), (w - 1, card.shape[0] - 1), (70, 70, 70), 1)
        return card

    def _draw_warning_box(self, image: np.ndarray, line1: str, line2: str) -> np.ndarray:
        """Helper to overlay a centered warning callout."""
        annotated = image.copy()
        h, w = annotated.shape[:2]
        bw, bh = 360, 70
        x1, y1 = (w - bw) // 2, (h - bh) // 2
        x2, y2 = x1 + bw, y1 + bh

        overlay = annotated.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.75, annotated, 0.25, 0, annotated)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), config.COLOR_YELLOW, 2)

        cv2.putText(annotated, line1, (x1 + 25, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, config.COLOR_YELLOW, 2, cv2.LINE_AA)
        cv2.putText(annotated, line2, (x1 + 30, y1 + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.45, config.COLOR_WHITE, 1, cv2.LINE_AA)
        return annotated

    # =========================================================================
    # Button & Event Handlers
    # =========================================================================
    def _toggle_camera(self):
        """Starts or stops the webcam hardware stream."""
        if not self.is_camera_running:
            if self.camera.start():
                self.is_camera_running = True
                self.btn_camera.configure(text="Stop Camera Feed", fg_color="#c92a2a", hover_color="#a61e1e")
                self.set_status("🟢 Camera connected and streaming.")
            else:
                self.set_status("❌ Failed to initialize webcam. Check if another app is using it.")
        else:
            self.camera.release()
            self.is_camera_running = False
            self.btn_camera.configure(text="Start Camera Feed", fg_color="#2b8a3e", hover_color="#237032")
            self.lbl_video.configure(image=None, text="Webcam Feed Inactive\n\nClick 'Start Camera Feed' on the left to begin.")
            self.lbl_header_info.configure(text="🔴 Camera: Offline")
            self.lbl_fps_badge.configure(text="FPS: 0.0")
            self.set_status("Camera stream stopped.")

    def _action_capture_bg(self):
        """Captures a clean 30-frame background using BackgroundManager."""
        if not self.is_camera_running:
            self.set_status("⚠️ Cannot capture background: Please start the camera first.")
            return

        self.set_status("Capturing 30 frames for clean background... Please step out of view!")
        self.update_idletasks()

        captured = self.bg_manager.capture_background(self.camera, num_frames=config.BG_CAPTURE_FRAMES)
        if captured:
            self.lbl_bg_status.configure(text="Status: Captured & Ready", text_color="#51cf66")
            self.set_status("✅ Background captured successfully! Invisibility effect is ready.")
        else:
            self.set_status("❌ Background capture failed.")

    def _action_calibrate_roi(self):
        """Samples the fabric in the central ROI and registers a custom calibrated preset."""
        if not self.is_camera_running:
            self.set_status("⚠️ Please start the camera first.")
            return

        ret, frame = self.camera.read_frame(flip_horizontal=True)
        if not ret or frame is None:
            self.set_status("⚠️ Could not grab frame for calibration.")
            return

        calib_preset = self.calibrator.calibrate(frame)
        if calib_preset:
            self.detector.register_custom_preset(calib_preset)
            self.opt_color.set("Custom Calibrated")
            self.lbl_calib_info.configure(
                text=f"✅ {calib_preset['display_name']}\nBounds: H:[{calib_preset['telemetry']['h_range'][0]}-{calib_preset['telemetry']['h_range'][1]}], S:[{calib_preset['telemetry']['s_range'][0]}-{calib_preset['telemetry']['s_range'][1]}]",
                text_color="#51cf66",
            )
            self.set_status(f"🎯 Auto-Calibration Successful! Activated custom preset.")

    def _on_color_selected(self, choice: str):
        """Dropdown event handler for selecting cloak color preset."""
        mapping = {
            "Blue Cloak": "blue",
            "Red Cloak": "red",
            "Green Cloak": "green",
            "Yellow Cloak": "yellow",
            "Custom Calibrated": "custom",
        }
        color_key = mapping.get(choice, "blue")
        if color_key == "custom" and not self.detector.has_custom_calib:
            self.set_status("⚠️ Custom color not yet calibrated. Click 'Auto-Calibrate from ROI'.")
            self.opt_color.set(self.detector.get_display_name())
            return

        self.detector.set_color(color_key)
        self.set_status(f"Switched detection to: {self.detector.get_display_name()}")

    def _on_toggle_invisibility(self):
        """Toggles the invisibility rendering effect."""
        self.invisibility_enabled = self.sw_invisibility.get() == 1
        status = "ENABLED" if self.invisibility_enabled else "DISABLED"
        self.set_status(f"Invisibility Effect: {status}")

    def _on_toggle_ai(self):
        """Toggles optional AI Person Segmentation."""
        self.ai_enabled = self.sw_ai.get() == 1
        if self.ai_enabled:
            if self.ai_segmenter.is_available():
                self.set_status("🤖 AI Person Segmentation ENABLED (Hybrid Mode: P ∩ C).")
            else:
                self.set_status("⚠️ AI model unavailable. Falling back to OpenCV baseline.")
        else:
            self.set_status("⚡ AI Person Segmentation DISABLED (OpenCV Baseline Active).")

    def _on_toggle_feathering(self):
        """Toggles soft edge alpha feathering."""
        enabled = self.engine.toggle_feathering()
        self.set_status(f"Edge Feathering / Alpha Blending: {'ENABLED' if enabled else 'DISABLED'}")

    def _on_toggle_temporal(self):
        """Toggles temporal anti-flicker smoothing."""
        enabled = self.detector.toggle_temporal_smoothing()
        self.set_status(f"Temporal Anti-Flicker (EMA): {'ENABLED' if enabled else 'DISABLED'}")

    def _on_view_mode_changed(self, value: str):
        """Segmented button event handler for changing viewport layout."""
        self.active_view_mode = value
        self.set_status(f"Viewport Mode: {value}")

    def _action_take_screenshot(self):
        """Exports the current rendered frame to the output/screenshots/ folder."""
        if not hasattr(self, "_last_rendered_frame") or self._last_rendered_frame is None:
            self.set_status("⚠️ No active frame to capture.")
            return

        try:
            path = self.recorder.save_screenshot(self._last_rendered_frame)
            filename = os.path.basename(path)
            self.set_status(f"📸 Screenshot saved: {filename} in output/screenshots/")
        except Exception as e:
            self.set_status(f"❌ Failed to save screenshot: {e}")

    def _action_toggle_recording(self):
        """Starts or stops MP4 video recording."""
        if not self.is_camera_running:
            self.set_status("⚠️ Please start the camera feed before recording.")
            return

        if not self.recorder.is_recording():
            path = self.recorder.start_recording(
                width=config.FRAME_WIDTH,
                height=config.FRAME_HEIGHT,
                fps=config.FPS_TARGET,
            )
            self.btn_record.configure(text="⏹️ Stop Recording (00:00)", fg_color="#e03131", hover_color="#c92a2a")
            self.set_status(f"🔴 Recording started: {os.path.basename(path)}")
        else:
            path, frames, dur = self.recorder.stop_recording()
            self.btn_record.configure(text="🔴 Start MP4 Recording", fg_color="#c92a2a", hover_color="#a61e1e")
            self.set_status(f"✅ Video saved: {os.path.basename(path)} ({frames} frames, {dur:.1f}s)")

    def _action_open_folder(self):
        """Opens the output directory in the operating system's file manager."""
        folder_path = os.path.abspath(config.OUTPUT_DIR)
        os.makedirs(folder_path, exist_ok=True)
        try:
            if platform.system() == "Windows":
                os.startfile(folder_path)
            elif platform.system() == "Darwin":
                subprocess.Popen(["open", folder_path])
            else:
                subprocess.Popen(["xdg-open", folder_path])
            self.set_status(f"📁 Opened output folder: {folder_path}")
        except Exception as e:
            self.set_status(f"❌ Could not open folder: {e}")

    def set_status(self, message: str):
        """Updates the bottom status toast message."""
        self.status_message = message
        self.lbl_status.configure(text=message)

    def _on_close(self):
        """Graceful shutdown: releases hardware, finalizes recordings, and destroys window."""
        print("[INFO] Shutting down Smart Invisible Cloak Application...")
        if self.recorder.is_recording():
            self.recorder.stop_recording()
        if self.camera.is_running():
            self.camera.release()
        self.destroy()


def launch_app():
    """Entry point for launching the Desktop GUI."""
    app = SmartInvisibleCloakApp()
    app.mainloop()


if __name__ == "__main__":
    launch_app()
