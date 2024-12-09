import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

# Inisialisasi utilitas MediaPipe untuk menggambar landmark tangan
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Fungsi untuk menutup jendela video dan menghentikan program
def close_window():
    cap.release()
    cv2.destroyAllWindows()
    exit()

# Membuka kamera (0 = kamera default)
cap = cv2.VideoCapture(0) 

# Inisialisasi deteksi tangan MediaPipe
hands = mp_hands.Hands()

# Loop utama untuk membaca frame dari kamera
while True:
    ret, frame = cap.read()  # Membaca frame dari kamera
    if not ret:
        break  # Keluar dari loop jika tidak ada frame yang terbaca

    # Mengubah frame dari BGR (OpenCV) ke RGB (MediaPipe)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Memproses gambar untuk mendeteksi tangan
    results = hands.process(image_rgb)

    # Mengembalikan frame ke format BGR agar dapat ditampilkan di OpenCV
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    # Jika ada tangan yang terdeteksi
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Menggambar landmark tangan dan koneksi di frame
            mp_drawing.draw_landmarks(
                image_bgr,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # Menyimpan koordinat landmark dalam array NumPy
            landmark_array = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])

            # Meratakan array menjadi vektor satu dimensi
            landmark_array = landmark_array.flatten()

            # Normalisasi data landmark untuk mengurangi noise
            landmark_array = (landmark_array - np.mean(landmark_array)) / np.std(landmark_array)

            # Mendapatkan ukuran frame
            h, w, c = image_bgr.shape

            # Menghitung koordinat piksel dari landmark tangan
            landmark_coords = np.array([[int(lm.x * w), int(lm.y * h)] for lm in hand_landmarks.landmark])

            # Menentukan batas kotak pembatas (bounding box)
            x_min, y_min = np.min(landmark_coords, axis=0)
            x_max, y_max = np.max(landmark_coords, axis=0)

            # Menggambar kotak pembatas di sekitar tangan
            cv2.rectangle(image_bgr, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    # Menampilkan frame dengan tangan yang terdeteksi
    cv2.imshow("Deteksi Bahasa Isyarat", image_bgr)

    # Menutup jendela jika tombol 'q' ditekan
    if cv2.waitKey(1) & 0xFF == ord('q'):
        close_window()

# Membersihkan semua jendela setelah loop selesai
cv2.destroyAllWindows()
