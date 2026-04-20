from torch.cuda import device
from ultralytics import YOLO
import torch


def main():
    # Вся логика находится внутри функции
    model = YOLO('yolov8n-seg.pt')
    data_path = 'data.yaml'

    # Запуск обучения
    results = model.train(data=data_path, epochs=100, imgsz=640, device='cuda')
    print("Обучение завершено!")


if __name__ == '__main__':
    main()