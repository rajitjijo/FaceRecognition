# 🕵️‍♂️ Real-Time Face Blurring using InsightFace and OpenCV

This project captures webcam video, detects faces using [InsightFace](https://github.com/deepinsight/insightface)'s RetinaFace detector, and **blurs them in real-time** using OpenCV. It's a lightweight privacy-focused tool that ensures no faces are visibly exposed during webcam usage or video recording.

---

## 📸 Demo

https://user-images.githubusercontent.com/your-username/demo.gif  
*(Optional: Add a demo GIF here)*

---

## 🚀 Features

- Real-time webcam stream processing
- Accurate face detection using `buffalo_l` model from InsightFace
- Gaussian blur applied to detected face regions
- GPU acceleration support (via CUDA / TensorRT if configured)
- Clean, minimal code — easy to extend (e.g., recognition, masking)
