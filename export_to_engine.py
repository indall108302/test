from ultralytics import YOLO
import os

def convert_model(model_path, device=0):
    if not os.path.exists(model_path):
        print(f"Ошибка: Файл {model_path} не найден!")
        return

    model = YOLO(model_path)

    export_path = model.export(
        format='engine', 
        device=device, 
        half=True, 
        simplify=True,
        workspace=4 # ГБ
    )

if __name__ == "__main__":
    MODEL_PATH = 'runs/segment/train2/weights/best.pt'
    convert_model(MODEL_PATH)
