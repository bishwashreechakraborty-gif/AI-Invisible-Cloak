# 🪄 AI Invisible Cloak

A real-time computer vision project that creates an "invisible cloak" effect using Python, OpenCV and AI-based person segmentation.

The system captures the background first and then replaces the selected cloak/person region with the previously captured background, creating the illusion that the person has disappeared.

## ✨ Features

- Real-time webcam processing
- Background capture
- HSV-based cloak color detection
- Red and Blue cloak presets
- Custom color calibration
- Mask refinement and smoothing
- AI-based person segmentation
- Invisible effect
- Anti-flicker processing
- Edge feathering
- GUI interface
- Screenshot and video recording
- AI Debug View
- Automated tests for all 7 stages

## 🛠️ Technologies

- Python
- OpenCV
- NumPy
- MediaPipe
- CustomTkinter
- Pillow

## 📁 Project Structure

```text
AI Invisible Cloak/
│
├── main.py
├── config.py
├── camera.py
├── background.py
├── calibrator.py
├── color_detector.py
├── invisibility_engine.py
├── ai_segmenter.py
├── gui_app.py
├── recorder.py
│
├── test_stage1.py
├── test_stage2.py
├── test_stage3.py
├── test_stage4.py
├── test_stage5.py
├── test_stage6.py
├── test_stage7.py
│
├── requirements.txt
└── README.md
```

## 🚀 Installation

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd "AI Invisible Cloak"
```

### 2. Install the required packages

```bash
pip install -r requirements.txt
```

### 3. Run the application

```bash
python main.py
```

## 🎥 How It Works

1. Start the webcam.
2. Capture the background without the person.
3. Select a cloak color or use color calibration.
4. Wear the cloak and stand in front of the camera.
5. The system detects the cloak/person.
6. The detected region is replaced with the captured background.
7. The result is displayed in real time.

## 🔬 Development Stages

- **Stage 1:** Webcam + Background Capture
- **Stage 2:** HSV Cloak Detection
- **Stage 3:** Invisible Effect
- **Stage 4:** Mask Improvement
- **Stage 5:** Multiple Colors + Calibration
- **Stage 6:** GUI + Recording
- **Stage 7:** AI Person Segmentation + Hybrid Masking

## 🧪 Testing

Each stage has its own test file:

```bash
python test_stage1.py
python test_stage2.py
python test_stage3.py
python test_stage4.py
python test_stage5.py
python test_stage6.py
python test_stage7.py
```

## ⚠️ Limitations

- Works best with good lighting.
- A relatively static background gives better results.
- Similar-colored objects may affect color detection.
- AI segmentation may not be perfect with unusual poses or loose clothing.
- Performance depends on the webcam and computer.

## 📌 Project Purpose

This project was created to learn and demonstrate practical applications of computer vision, image segmentation, masking, background replacement and real-time AI processing.
