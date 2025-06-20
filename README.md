# 😄 Facial Emotion Recognition with CNN & MediaPipe

This project implements a **real-time facial emotion recognition system** using a **Convolutional Neural Network (CNN)** trained on the **FER-2013** dataset and face mesh detection using **MediaPipe**. It captures webcam input, detects facial landmarks, and predicts emotional states like Happy, Sad, Angry, etc., on-the-fly.

---

## 🧠 What It Does

✅ Detects faces using MediaPipe  
✅ Applies 468-point face mesh with light styling  
✅ Predicts emotions like 😠 😢 😍 😮 😐 from facial expressions  
✅ Uses a CNN trained on **FER-2013 dataset**  
✅ Runs live via webcam (OpenCV) — No images saved!


---

## 📦 Dataset Used

- **FER-2013 (Facial Expression Recognition 2013)**
- Download from: https://www.kaggle.com/datasets/msambare/fer2013
- Format: Pre-separated folder structure with `train/` and `validation/` directories.
- Each should contain 7 folders: angry/, disgust/, fear/, happy/, neutral/, sad/, surprise/

### workflow
> - pip install -r requirements.txt
> - facial_emo_recognition/
> - └── dataset/
>    - ├── train/
>    - └── validation/
> - python train_emo_model.py
>  run
> - python main.py



