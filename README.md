# Deepfake Detection using AI & VGG19

A **Flask-based AI web application** that detects whether an image is **real** or **fake** using a **pre-trained VGG19 deep learning model**. Leveraging AI, the app provides fast and accurate predictions for uploaded images.

---

## 🚀 Features

* **AI-powered detection** using a pre-trained **VGG19 convolutional neural network**.
* Web interface for easy image upload.
* Predicts **Real** vs **Fake** images using deep learning.
* Supports `.jpg`, `.jpeg`, `.png` image formats.
* Lightweight Flask app with JSON-based predictions.
* Logs all prediction steps for debugging.

---

## 🗂 Project Structure

```
deepfake_detection_usingvgg19/
│
├── app.py                     # Flask application with AI integration
├── deepfake_detection4_vgg19_with_plots.h5  # Pre-trained VGG19 AI model
├── static/
│   └── uploads/               # Uploaded images storage
├── templates/
│   ├── index.html             # Main upload page
│   └── thankyou.html          # Thank you page
└── .gitignore
```

---

## ⚙ Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/deepfake_detection_usingvgg19.git
cd deepfake_detection_usingvgg19
```

2. Create and activate a virtual environment:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

> **Note:** Required libraries include `tensorflow`, `keras`, `flask`, and `numpy`.

---

## 🖥 Usage

1. Run the Flask application:

```bash
python app.py
```

2. Open your browser at:

```
http://127.0.0.1:5000/
```

3. Upload an image to get an AI-based **Real/Fake prediction**.

---

## 🔍 Endpoints

* `/` - Home page for uploading images.
* `/detect` - POST endpoint for submitting images and receiving AI predictions.
* `/thankyou` - Thank you page after processing.
* `/exit` - Redirects to the thank you page.

---

## 🧠 How AI Works

1. The uploaded image is **preprocessed** to `224x224` pixels to match VGG19 input requirements.
2. Image array is normalized using `preprocess_input` for VGG19.
3. Pre-trained **VGG19 CNN model** predicts whether the image is **Real** or **Fake**.
4. Result is returned as JSON and displayed to the user.

---

## ⚠ Notes

* Large model files (`.h5`) are **not recommended for GitHub**; use `.gitignore` or **Git LFS**.
* Only allowed formats: `.jpg`, `.jpeg`, `.png`.
* Upload folder `static/uploads/` is created automatically if missing.

---

## 📄 License

This project is licensed under the MIT License.

