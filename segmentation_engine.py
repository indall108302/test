import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict

class SegmentationEngine:
    def __init__(self, model_path='best.pt'):
        # Загружаем модель (используем GPU если доступно)
        self.model = YOLO(model_path)
        self.reset()

    def reset(self):
        """Сброс состояния трекинга и счетчиков"""
        self.seen_objects = {}  # track_id -> class_id
        self.class_totals = defaultdict(int)  # class_id -> count
        self.track_history = defaultdict(lambda: [])

    def process_frame(self, frame, conf=0.5, iou=0.7, tracker="botsort.yaml"):
        """
        Обработка одного кадра: детекция, трекинг и подсчет.
        """
        # Инференс
        results = self.model.track(
            frame, 
            persist=True, 
            conf=conf, 
            iou=iou, 
            tracker=tracker, 
            verbose=False
        )

        annotated_frame = results[0].plot()

        if results[0].boxes.id is not None:
            track_ids = results[0].boxes.id.int().cpu().tolist()
            class_ids = results[0].boxes.cls.int().cpu().tolist()
            
            for track_id, cls_id in zip(track_ids, class_ids):
                # Если объект новый - учитываем его
                if track_id not in self.seen_objects:
                    self.seen_objects[track_id] = cls_id
                    self.class_totals[cls_id] += 1

        return annotated_frame

    def get_stats(self):
        """Возвращает текущую статистику подсчета"""
        return dict(self.class_totals)

    def get_class_names(self):
        """Возвращает маппинг ID класса -> Имя"""
        return self.model.names
