import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

# Inisialisasi utilitas MediaPipe untuk menggambar landmark tangan
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
                mp_hands.HAND_CONNECTIONS
            )

            landmark_array = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])

            landmark_array = landmark_array.flatten()

            landmark_array = (landmark_array - np.mean(landmark_array)) / np.std(landmark_array)

            h, w, c = image_bgr.shape

            landmark_coords = np.array([[int(lm.x * w), int(lm.y * h)] for lm in hand_landmarks.landmark])

            x_min, y_min = np.min(landmark_coords, axis=0)
            x_max, y_max = np.max(landmark_coords, axis=0)

            cv2.rectangle(image_bgr, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    cv2.imshow("Deteksi Bahasa Isyarat", image_bgr)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        close_window()

cv2.destroyAllWindows()
