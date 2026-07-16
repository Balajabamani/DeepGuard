# DeepGuard

## AI-Powered Deepfake Detection System with Forensic Visualization

DeepGuard is an AI-powered deepfake image detection application developed using **EfficientNetB4**, **MTCNN**, and **Grad-CAM**. It detects manipulated facial images and provides visual explanations through forensic heatmaps.

---

## Features

- Deepfake image detection using EfficientNetB4
- Face detection using MTCNN
- Grad-CAM forensic heatmap visualization
- Five-level confidence verdict system
- Upload image or webcam capture
- PDF and CSV forensic report generation
- Streamlit web interface

---

## Technologies Used

- Python
- TensorFlow / Keras
- Streamlit
- OpenCV
- MTCNN
- NumPy

---

## Project Structure

```
DeepGuard/
│── app.py
│── models/
│── DeepGuard_Presentation.pptx
│── Deepfake architecture.png
│── .gitignore
```

---

## Installation

Clone the repository

```bash
git clone https://github.com/Balajabamani/DeepGuard.git
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run

```bash
streamlit run app.py
```

---

## Model

- EfficientNetB4 (Transfer Learning)
- Image Size: 224 × 224
- Face Detection: MTCNN
- Explainability: Grad-CAM

---

## Author

**Balajabamani D**

M.Sc. Artificial Intelligence & Cyber Security
