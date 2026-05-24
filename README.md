# 🇮🇳 Intelligent Traffic Sign Classification System (Explainable ADAS)
---

#  Project Overview

This repository contains a deep learning-powered **Advanced Driver Assistance System (ADAS)** designed for **traffic sign recognition in Indian roadside environments**.

The project combines:
- Computer Vision
- Explainable AI (XAI)
- Real-Time Inference
- Transfer Learning
- Multi-Domain Learning
- Voice-Based Driver Alerts

to simulate an intelligent transportation assistance system capable of identifying and interpreting roadside traffic signs under complex real-world conditions.

Unlike standard traffic sign classifiers trained only on benchmark datasets, this system investigates the challenges of **domain shift** between European and Indian traffic sign environments through a structured experimental pipeline.

---

#  Key Features

##  Deep Learning Classification Engine
- Custom-trained **ResNet18** architecture
- Multi-class traffic sign recognition
- Optimized for Indian roadside conditions
- Trained using PyTorch

---

##  Explainable AI with Grad-CAM
- Integrated Grad-CAM visualization pipeline
- Highlights model attention regions
- Verifies prediction reliability
- Reduces black-box uncertainty

The Grad-CAM module hooks into:
```python
layer4[-1]
```

to generate class-specific activation heatmaps.

---

##  Real-Time Voice Alert System
- Offline text-to-speech alerts using `pyttsx3`
- Multi-threaded audio execution
- Non-blocking UI pipeline
- Context-aware driver warnings

Example:
```text
"Pedestrian crossing detected. Watch out for people."
```

---

##  Streamlit Dashboard
- Interactive deployment interface
- Real-time prediction visualization
- Grad-CAM heatmap display
- User-friendly diagnostic dashboard

---

#  System Architecture

```text
Input Road Scene
        ↓
Image Preprocessing
        ↓
ResNet18 Classification Engine
        ↓
Prediction Layer
   ┌───────────────┬────────────────┐
   ↓               ↓
Grad-CAM       Voice Alert Engine
   ↓               ↓
Heatmap        Spoken Driver Alert
   └───────────────┬────────────────┘
                   ↓
           Streamlit Dashboard
```

---

#  Experimental Framework

To evaluate robustness and domain adaptation capabilities, the system was trained through a structured multi-stage experimentation pipeline.

---

## Experiment 1 — Baseline Training

| Parameter | Value |
|---|---|
| Dataset | GTSRB |
| Strategy | Training from Scratch |
| Objective | Benchmark Baseline |
| Result | 99.97% Accuracy |

---

## Experiment 2 — Cross-Domain Evaluation

| Parameter | Value |
|---|---|
| Train Dataset | GTSRB |
| Test Dataset | Indian Traffic Signs |
| Objective | Domain Shift Analysis |
| Result | 2.02% Accuracy |

### Key Insight
Direct deployment of European-trained traffic sign models fails catastrophically in Indian roadside environments due to severe domain mismatch.

---

## Experiment 3 — Transfer Learning

| Parameter | Value |
|---|---|
| Dataset | Indian Traffic Signs |
| Strategy | Fine-Tuning |
| Learning Rate | $10^{-4}$ |
| Result | 80.59% Accuracy |

### Key Insight
Transfer learning significantly improves cross-domain performance but still struggles with environmental complexity and background clutter.

---

## Experiment 4 — Native Indian Training

| Parameter | Value |
|---|---|
| Dataset | Indian Traffic Signs |
| Strategy | Native Training |
| Learning Rate | $10^{-4}$ |
| Result | **91.15% Test Accuracy** |

### Engineering Insight
Training directly on Indian roadside environments forces convolutional filters to learn localized visual features, improving robustness against:
- cluttered backgrounds,
- lighting variability,
- road-side noise,
- non-standard sign conditions.

---

#  Domain Shift Analysis

This project strongly demonstrates the impact of **domain shift** in computer vision systems.

A model trained on:
- clean European roads,
- standardized sign structures,
- controlled image conditions

fails when deployed on:
- noisy Indian roads,
- inconsistent viewpoints,
- environmental clutter,
- varied lighting conditions.

This highlights the importance of:
- localized datasets,
- domain adaptation,
- real-world training distributions.

---

#  Explainable AI (XAI)

The project integrates **Grad-CAM** to interpret CNN predictions.

Grad-CAM helps verify:
- whether the model focuses on the traffic sign,
- or incorrectly attends to background regions.

This improves:
- model transparency,
- debugging,
- trustworthiness,
- deployment reliability.

---

#  Asynchronous Voice Alert Pipeline

To prevent UI lag during speech generation, the audio system uses:
- background daemon threads,
- asynchronous execution,
- decoupled speech synthesis.

This allows:
- smooth rendering,
- continuous inference,
- non-blocking voice playback.

---

#  Tech Stack

| Component | Technology |
|---|---|
| Deep Learning | PyTorch |
| Computer Vision | OpenCV |
| Dashboard | Streamlit |
| Explainability | Grad-CAM |
| Voice Alerts | pyttsx3 |
| Visualization | Matplotlib |
| Data Processing | NumPy, Pandas |
| Training Platform | Kaggle |

---

#  Core Model Architecture

The primary architecture explored in this project:

- ResNet18
- Transfer Learning
- Fine-Tuning Strategies
- Native Domain Training

Additional experiments:
- Cross-domain evaluation
- Multi-domain training
- Dataset harmonization

---

#  Planned Interface

The Streamlit interface will support:
- Image upload
- Random test generation
- Live webcam prediction
- Grad-CAM visualization
- Voice alert playback
- Real-time classification dashboard

---

#  Installation

## Clone Repository

```bash
git clone https://github.com/Anamika74/traffic-sign-recognition.git

cd traffic-sign-recognition
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶ Run Streamlit App

```bash
streamlit run app/streamlit_app.py
```

---

#  Future Improvements

- YOLO-based traffic sign detection
- Video stream tracking
- Multilingual voice alerts
- GPS-aware contextual warnings

---

#  Key Learning Outcomes

This project explores:
- CNN architectures
- Transfer Learning
- Domain Adaptation
- Cross-Domain Generalization
- Explainable AI
- Real-Time Inference
- Streamlit Deployment
- Multi-Threading
- Dataset Engineering

---

#  Conclusion

This project demonstrates how deep learning systems trained on ideal benchmark datasets often fail in real-world deployment environments due to domain shift.

Through structured experimentation, explainable AI integration, and deployment-focused engineering, the system evolves from a standard image classifier into a practical ADAS-oriented computer vision pipeline tailored for Indian roadside conditions.

---

