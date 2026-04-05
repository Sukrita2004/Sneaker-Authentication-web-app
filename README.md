Sneaker Authentication web application

A web-based AI application that uses Deep Learning (CNNs) and Computer Vision (SIFT) to authenticate Adidas sneakers. It detects the shoe model, extracts primary colors, analyzes logo accuracy, and classifies the shoe as Genuine or Counterfeit.

Technology use:
1. Backend & AI: Python, Flask, TensorFlow (MobileNetV2 & Custom CNN), OpenCV (SIFT), Rembg.

2. Frontend: HTML5, Tailwind CSS.

Setup (Local)

1. Install Dependencies:
pip install flask tensorflow opencv-python numpy Pillow rembg easyocr scipy

3. Add Required Files:
Place your custom AI model (supervised_shoe_model.h5) directly in the root folder.
Ensure all your reference logo images (e.g., default_logo.jpg, adidas_samba_logo.jpg) are placed directly in the root folder right alongside app.py.

5. Run the App:

python3 app.py

Open http://127.0.0.1:5000 in your browser.

Sukrita Chaoyong
