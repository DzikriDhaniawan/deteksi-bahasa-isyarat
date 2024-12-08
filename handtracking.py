import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

def close_window():
    cap.release()
    cv2.destroyAllWindows()
    exit()


cap = cv2.VideoCapture(0)
hands = mp_hands.Hands()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image_bgr,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS)

            landmark_array = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
            landmark_array = landmark_array.flatten()

            landmark_array = (landmark_array - np.mean(landmark_array)) / np.std(landmark_array)

    cv2.imshow("Hand Gesture Recognition", image_bgr)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        close_window()
