import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict

# Загружаем обученную модель
model = YOLO('runs/segment/train2/weights/best.pt')

# Открываем видеофайл
video_path = 0
cap = cv2.VideoCapture(video_path)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 30.0, (1920, 1080))

# Словарь для хранения "хвостов" (траекторий) для каждого объекта
track_history = defaultdict(lambda: [])

# Глобальный счётчик уникальных объектов
seen_objects = {}  # track_id -> class_id
class_totals = defaultdict(int)  # class_id -> count

if not cap.isOpened():
    print(f"Ошибка: не удалось открыть видео по пути {video_path}")
    exit()

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    results = model.track(frame, persist=True, tracker="bytetrack.yaml")

    if results[0].boxes.id is not None and results[0].masks is not None:
        masks = results[0].masks.xy
        track_ids = results[0].boxes.id.int().cpu().tolist()
        class_ids = results[0].boxes.cls.int().cpu().tolist()

        for mask, track_id, cls_id in zip(masks, track_ids, class_ids):
            # === если впервые видим объект с таким ID, учитываем его класс ===
            if track_id not in seen_objects:
                seen_objects[track_id] = cls_id
                class_totals[cls_id] += 1

            # Рисуем контур маски
            points = np.array(mask, dtype=np.int32)
            cv2.polylines(frame, [points], isClosed=True, color=(0, 255, 0), thickness=2)

            # Пишем ID и класс над объектом
            text_pos = (points[0][0], points[0][1] - 10)
            cv2.putText(
                frame,
                f"ID: {track_id} Class: {cls_id}",
                text_pos,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

        # === выводим общую статистику (за всё видео) ===
        y0 = 30
        for cls_id, total in class_totals.items():
            cv2.putText(
                frame,
                f"Class {cls_id}: {total}",
                (10, y0),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )
            y0 += 30

    cv2.imshow("YOLOv8 Segmentation Tracking", frame)
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

# Итоговый вывод
print("📊 Итоговая статистика по видео:")
for cls_id, total in class_totals.items():
    print(f"Class {cls_id}: {total}")
