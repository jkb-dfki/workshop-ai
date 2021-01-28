# System Imports
import math
from enum import Enum

# Third-Party Imports
import cv2
import numpy as np
import mediapipe as mp



capture = cv2.VideoCapture(0)

model = mp.solutions.hands.Hands(
    min_detection_confidence=0.6,
    min_tracking_confidence=0.5)

while capture.isOpened():

    readOk, imageInput = capture.read()

    if not readOk:
      break

    image = cv2.cvtColor(imageInput, cv2.COLOR_BGR2RGB)
    results = model.process(image)

    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:
            
            print(hand_landmarks)
            print(hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_TIP])
            mp.solutions.drawing_utils.draw_landmarks(image, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)

    imageOutput = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    cv2.imshow('', imageOutput)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

model.close()
capture.release()
