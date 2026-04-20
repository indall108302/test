import cv2
import os

# путь к видео
video_path = "video_3.mp4"
# папка для сохранения кадров
output_dir = "frames"
os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)

saved = 0
current_frame = 0
step = 5  # сначала каждые 5 кадров
limit_first = 32  # сколько кадров собрать с шагом 5

while True:
    ret, frame = cap.read()
    if not ret:
        break  # конец видео

    # сохраняем кадр, если он попадает в нужный шаг
    if saved < limit_first:
        if current_frame % step == 0:
            filename = os.path.join(output_dir, f"frame_00000{current_frame}.jpg")
            cv2.imwrite(filename, frame)
            saved += 1
            if saved == limit_first:
                step = 10  # после 30 кадров переходим на шаг 10
    else:
        if current_frame % step == 0:
            filename = os.path.join(output_dir, f"frame_00000{current_frame}.jpg")
            cv2.imwrite(filename, frame)

    current_frame += 1

cap.release()
print(f"Кадры сохранены в папку '{output_dir}'")
