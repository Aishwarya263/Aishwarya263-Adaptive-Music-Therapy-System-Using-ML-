# -*- coding: utf-8 -*-
"""Untitled.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1nzttD64arSORCCwJnNPHKboZgFMUChqs
"""



from google.colab import drive
drive.mount('/content/drive')

!pip install deepface

import cv2
import matplotlib.pyplot as plt
from deepface import DeepFace

def capture_image():
    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Couldn't access the webcam")
        return None

    # Capture frame-by-frame
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("Error: Couldn't capture frame")
        return None

    # Convert the frame from BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    return frame_rgb

# Capture image from webcam
img1 = capture_image()

# Analyze emotions in the captured image if img1 is not None
if img1 is not None:
    results = DeepFace.analyze(img1, actions=['emotion'], enforce_detection=False)
    for result in results:
        emotions = result['emotion']
        sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)
        for emotion, score in sorted_emotions:
            print(f"{emotion}: {score}")

    # Display the captured image
    plt.imshow(img1)
    plt.axis('off')
    plt.show()
else:
    print("Error: Couldn't capture image from webcam")