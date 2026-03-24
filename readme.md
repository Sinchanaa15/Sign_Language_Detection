# Real-Time Sign Language Detection using MediaPipe, OpenCV and Random Forest

## Abstract

This project implements a real-time sign language recognition system using a standard webcam. The goal is to recognize a fixed vocabulary of hand signs (HELLO, YES, NO, THANK YOU, BEAUTIFUL, CAREFUL, FIGHT, GREAT, NOW, SEE, TALK, US) and display the predicted label live on the screen, along with a sentence-building mode and prediction history.

The system uses MediaPipe Hands to extract 3D hand landmarks from each video frame, transforms them into a 63-dimensional feature vector (21 landmarks × (x, y, z)), and trains a classical machine learning model (RandomForestClassifier) on these features. The trained model is served via a FastAPI backend, while a web-based frontend (HTML + JavaScript) uses the browser’s webcam, sends frames to the backend, and renders live predictions and UI elements.

This approach avoids heavy deep learning models and focuses on efficient landmark-based features, making the system lightweight and suitable for real-time execution on a laptop.

---

## Frameworks, Libraries and Code Base

### Main frameworks and libraries

- **Python 3.10+**
- **MediaPipe** – hand landmark detection
- **OpenCV** – webcam capture and visualization
- **scikit-learn** – RandomForest classification, metrics
- **FastAPI** – REST API for image → prediction
- **Uvicorn** – ASGI server for FastAPI
- **NumPy, Pandas** – data handling and feature processing
- **joblib** – model serialization
- **HTML + JavaScript** – frontend UI for live camera and predictions

### Code structure

- `scripts/collect_data.py`  
  Captures webcam frames, runs MediaPipe Hands, converts landmarks to a 63-dimensional feature vector, and appends labelled samples to `data/landmarks.csv` based on keyboard shortcuts.

- `scripts/train_classifier.py`  
  Loads `landmarks.csv`, performs preprocessing (feature scaling), splits into train/test sets, trains a RandomForestClassifier inside a scikit-learn Pipeline, evaluates metrics (accuracy, balanced accuracy, F1 scores, confusion matrix), saves the trained pipeline to `models/sign_model.pkl`, and writes metrics/reports into the `reports/` folder.

- `backend/app.py`  
  FastAPI application exposing an endpoint `/predict-image` that receives an image frame, extracts MediaPipe landmarks, applies the same preprocessing as during training, and returns the predicted sign label (and confidence) as JSON.

- `frontend/index.html`  
  Browser-based UI that:
  - Uses `getUserMedia` to access the webcam.
  - Periodically captures frames to a hidden canvas.
  - Sends frames to `http://127.0.0.1:8000/predict-image`.
  - Displays the current prediction, confidence, sentence mode (sequence of signs), prediction history panel, and simple settings (detection speed, confidence toggle).

### Attribution

This project conceptually builds on:

- MediaPipe Hands:  
  - Lugaresi et al., “MediaPipe: A Framework for Building Perception Pipelines” (Google AI)
- Random forests:  
  - Breiman, L., “Random Forests,” Machine Learning, 2001.

Code inspiration and references:

- Official MediaPipe Hands examples and documentation (Google).
- FastAPI documentation for building simple image-based APIs.
- Basic scikit-learn examples for RandomForest and Pipelines.

All additional external code snippets, if used, are cited in code comments or this README.

---

## Dependencies

All Python dependencies are summarized in `requirements.txt`. Typical content:

```text
numpy
pandas
opencv-python
mediapipe
scikit-learn
fastapi
uvicorn
joblib
python-multipart
