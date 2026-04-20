import cv2
from ultralytics import YOLO
from collections import defaultdict
import numpy as np

model = YOLO('runs/segment/train2/weights/best.pt')
save_video = False
show_video = True
cap = cv2.VideoCapture(1)

# Словарь для хранения "хвостов" (траекторий) для каждого объекта
track_history = defaultdict(lambda: [])

# Глобальный счётчик уникальных объектов
seen_objects = {}  # track_id -> class_id
class_totals = defaultdict(int)  # class_id -> count

if not cap.isOpened():
    raise Exception("Error: Could not open video file.")

fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

if save_video:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter("output.mp4", fourcc, fps, (frame_height, frame_width))

seen_objects = {}               # track_id -> class_id
class_totals = defaultdict(int) # class_id -> count

# параметры ROI
margin_ratio = 0  # 0% от ширины слева и справа
x_margin = int(frame_height * margin_ratio)
roi_top = x_margin
roi_bottom = frame_height - x_margin

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

    # Ищем детальки и трекаем ботсортом(можно другим)
    results = model.track(frame, persist=True, iou=0.75, conf=0.50,
                          tracker="botsort.yaml", imgsz=640, verbose=False)
    annotated_frame = results[0].plot()

    # считаем объекты только внутри ROI
    if results[0].boxes.id is not None:
        track_ids = results[0].boxes.id.int().cpu().tolist()
        class_ids = results[0].boxes.cls.int().cpu().tolist()
        boxes = results[0].boxes.xyxy.cpu().numpy()  # xmin, ymin, xmax, ymax

        for track_id, cls_id, box in zip(track_ids, class_ids, boxes):
            x_center = (box[0] + box[2]) / 2
            if roi_bottom <= x_center <= roi_top:
                if track_id not in seen_objects:
                    seen_objects[track_id] = cls_id
                    class_totals[cls_id] += 1
    # обозначаем область подсчета
    dark_factor = 0

    # затемняем сверху и снизу
    annotated_frame[:roi_top, :] = (annotated_frame[:roi_top, :] * dark_factor).astype(np.uint8)
    annotated_frame[roi_bottom:, :] = (annotated_frame[roi_bottom:, :] * dark_factor).astype(np.uint8)

    # рисуем прямоугольник по границам ROI
    # cv2.rectangle(
    #     annotated_frame,
    #     (0, roi_top),
    #     (frame_width, roi_bottom),
    #     (0, 255, 255),
    #     2
    # )



    x0 = 30

    for cls_id, total in class_totals.items():
        text = f"Class {cls_id}: {total}"

        # создаем пустую "маску" для текста
        (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        mask = np.zeros((h + 10, w + 10, 3), dtype=np.uint8)

        # пишем текст на маске
        cv2.putText(mask, text, (5, h + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # поворачиваем на 90 градусов
        rotated = cv2.rotate(mask, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # вставляем в кадр (x, y = позиция левого верхнего угла)
        x0, y = x0, 0
        h_r, w_r = rotated.shape[:2]
        annotated_frame[y:y + h_r, x0:x0 + w_r] = rotated

        x0 += 30  # увеличиваем отступ, чтобы тексты не налезали


    if save_video:
        annotated_frame = cv2.rotate(annotated_frame, cv2.ROTATE_90_CLOCKWISE)
        out.write(annotated_frame)

    if show_video:
        resized_frame = cv2.resize(
            annotated_frame,
            (annotated_frame.shape[1] // 2, annotated_frame.shape[0] // 2)
        )
        cv2.imshow("Сегментация деталек", resized_frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
if save_video:
    out.release()
cv2.destroyAllWindows()

print("Итого")
for cls_id, total in class_totals.items():
    print(f"Class {cls_id}: {total}")
