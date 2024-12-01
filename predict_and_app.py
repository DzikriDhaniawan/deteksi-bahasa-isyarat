# predict_and_app.py
import cv2
import numpy as np
import tensorflow as tf
from preprocess_data import preprocess_image

model = tf.keras.models.load_model('sign_language_model.h5')

def predict_sign(image):
    image = preprocess_image(image)
    image = cv2.resize(image, (64, 64))
    image = image.reshape(1, 64, 64, 3) / 255.0
    prediction = model.predict(image)
    class_index = np.argmax(prediction)
    return chr(class_index + ord('A'))

# Implementasi aplikasi dengan webcam
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    hand_sign = predict_sign(frame)
    cv2.putText(frame, hand_sign, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.imshow('Sign Language Detector', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
