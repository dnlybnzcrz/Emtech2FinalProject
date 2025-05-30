# Object Detection with YOLOv3-Tiny

A Streamlit-based web application for real-time object detection using YOLOv3-Tiny model. This project was developed as part of the Final Exam for Emerging Technologies 2 in Computer Engineering (CPE 019-CPE32S2).

## 👥 Authors

- Diones, John Cedric N.
- Cruz, Daniel Y.

## 🚀 Features

- Upload images in various formats (JPG, PNG, JPEG)
- Real-time object detection using YOLOv3-Tiny model
- Specialized in detecting objects like:
  - Laptop
  - Refrigerator
  - Traffic Light
  - Bottle
  - Truck
  - Train
- Visual bounding box display with object labels
- Confidence-based detection (threshold: 0.5)

## 🛠️ Technologies Used

- Python
- Streamlit
- OpenCV (cv2)
- YOLOv3-Tiny
- NumPy

## 📋 Prerequisites

Before running this project, make sure you have all the required dependencies installed. You can install them using:

```bash
pip install -r requirements.txt
```

## 🔧 Project Structure

- `main_app.py` - Main Streamlit application file
- `yolov3-tiny.cfg` - YOLOv3-Tiny configuration file
- `yolov3-tiny.weights` - Pre-trained model weights
- `coco.names` - Class names for object detection
- `requirements.txt` - Python dependencies

## 🚀 How to Run

1. Clone this repository
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit application:
   ```bash
   streamlit run main_app.py
   ```
4. Upload an image and see the object detection in action!

## 📝 Note

This project uses the YOLOv3-Tiny model, which is a smaller and faster version of YOLOv3, optimized for speed while maintaining reasonable accuracy for object detection tasks.

## 🎓 Academic Context

This project was developed as part of the Final Examination for the Emerging Technologies 2 course in Computer Engineering at Technological Institute of the Philippines.
