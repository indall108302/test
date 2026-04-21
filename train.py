from ultralytics import YOLO
import torch

def main():
    # Загружаем "чистую" модель YOLOv11 для сегментации
    model = YOLO('yolo11n-seg.pt')
    
    # Запуск обучения на объединенном датасете (11 классов)
    # Используем 50 эпох, как и заказывали
    results = model.train(
        data='data.yaml', 
        epochs=50, 
        imgsz=640, 
        device='cuda' if torch.cuda.is_available() else 'cpu',
        project='runs/segment',
        name='combined_train'
    )
    print("Обучение успешно завершено!")

if __name__ == '__main__':
    main()