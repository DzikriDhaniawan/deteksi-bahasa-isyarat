import cv2
import os

camera = cv2.VideoCapture(0)

current_letter = 'A'
save_dir = f"dataset/{current_letter}"

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

img_counter = 0

while True:
    ret, frame = camera.read()
    if not ret:
        break

    cv2.imshow('Capture', frame)

    if cv2.waitKey(1) & 0xFF == ord('s'):
        img_name = f"{save_dir}/img_{img_counter}.png"
        cv2.imwrite(img_name, frame)
        print(f"{img_name} saved!")
        img_counter += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
