IP Webcam Gender Classification Using OpenCV and DNN Models 

:-  This Python project performs real-time gender classification using an IP webcam stream. It leverages deep learning models for face detection and gender prediction. The project uses OpenCV's DNN module and pre-trained models to detect faces and classify gender (Male or Female) with a majority voting mechanism to smooth predictions across frames.
## Tech Stack

**Client:** Python, Deep Learning (DNN), P Webcam
![cam](https://github.com/user-attachments/assets/1d7499bc-5eae-4f96-b195-fad45a5603eb)


## Features:

1. Face Detection: Utilizes a pre-trained Caffe face detection model to accurately detect faces in each frame.
2. Gender Classification: A separate pre-trained gender classification model predicts the gender based on detected faces.
3. Real-Time Processing: Processes live video feed from an IP webcam.
4. Smoothing Predictions: Implements a majority vote over the last few predictions for smoother results.
## How It Works:

1. Captures live video feed from the IP webcam.
2. Detects faces in the video frames.
3. Classifies the detected faces as Male or Female.
4. Displays the gender label and bounding box on the video stream.
5. Supports quitting the stream by pressing the 'q' key.
