import threading
import time
import cv2
import numpy as np
import torch
from torchvision import transforms

from MvImport.MvCameraControl_class import *

# Загружаем нейронную сеть и параметры для обработки
MODEL_PATH = 'paintdef_classifaer_49_0.999007683424187_0.9990076914439704.pt'
model = torch.load(MODEL_PATH, weights_only=False, map_location=torch.device('cpu'))
model.eval()
model.to("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=0.5, std=0.5)])
TARGET_FRAMES = [
    5, # точка старта обработки
    10, # исследуем боковую поверзность
    190,  # исследуем дно заготовки
    226 # выключение освещения
    ]

TARGET_CELLS = [10, 11, 12, 16, 17, 18, 19, 20, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
                33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51,
                52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 66, 67]
BOTTOM_CELLS = list(range(80))

SIZE_BOX = 150  # Размер квадратов, на которые будем нарезать кадр


# Функция для нарезки кадра и детектирования дефектов
def process_frame(frame):
    height, width, _ = frame.shape
    frame = cv2.copyMakeBorder(frame, 0, SIZE_BOX - height % SIZE_BOX, 0, SIZE_BOX - width % SIZE_BOX,
                               cv2.BORDER_CONSTANT, value=0)
    draw_frame = frame.copy()

    # Переменные для подсчета количества квадратов
    h_steps = frame.shape[0] // SIZE_BOX
    w_steps = frame.shape[1] // SIZE_BOX
    count = 0

    for i in range(w_steps):
        for j in range(h_steps):
            x0, y0 = i * SIZE_BOX, j * SIZE_BOX
            x1, y1 = (i + 1) * SIZE_BOX, (j + 1) * SIZE_BOX
            cell = frame[y0:y1, x0:x1]
            count += 1

            if count in TARGET_CELLS:
                x = transform(cell).unsqueeze(0).to("cuda" if torch.cuda.is_available() else "cpu")
                pred = torch.argmax(model(x), dim=1).item()
                color = (0, 255, 0) if pred == 0 else (0, 0, 255)
                cv2.rectangle(draw_frame, (x0, y0), (x1, y1), color, 2)

    return draw_frame


# Функция для записи видео
def create_video_writer(output_path, width, height, fps=10):
    return cv2.VideoWriter('capture.mp4', cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))


# Основная функция для захвата видео и обработки
def capture_and_process_video(camera, output_path, video_length_hours=1, cam=0, pData=0, nDataSize=0):
    stFrameInfo = MV_FRAME_OUT_INFO_EX()
    memset(byref(stFrameInfo), 0, sizeof(stFrameInfo))

    writer = create_video_writer(output_path, 1920, 1080)
    start_time = time.time()

    while True:
        ret = camera.MV_CC_GetOneFrameTimeout(pData, nDataSize, stFrameInfo, 1000)
        if ret == 0:
            img = np.asarray(pData).reshape(stFrameInfo.nHeight, stFrameInfo.nWidth, -1)
            img = cv2.cvtColor(cv2.resize(img, (1920, 1080)), cv2.COLOR_BGR2RGB)

            # Обработка кадра с подсветкой дефектов
            processed_frame = process_frame(img)
            writer.write(processed_frame)
            cv2.imshow("Processed View", processed_frame)

            # create new
            if time.time() - start_time > video_length_hours * 3600:
                writer.release()
                start_time = time.time()
                writer = create_video_writer(output_path, 1920, 1080)

            if cv2.waitKey(24) & 0xFF == 27:
                break

        else:
            print("Error capturing frame")

    writer.release()
    cv2.destroyAllWindows()


# Инициализация и запуск камеры
if __name__ == "__main__":
    cam = MvCamera()
    deviceList = MV_CC_DEVICE_INFO_LIST()
    tlayerType = MV_GIGE_DEVICE | MV_USB_DEVICE
    ret = MvCamera.MV_CC_EnumDevices(tlayerType, deviceList)

    if deviceList.nDeviceNum == 0:
        print("No device found")
        sys.exit()

    cam = MvCamera()
    stDeviceList = cast(deviceList.pDeviceInfo[0], POINTER(MV_CC_DEVICE_INFO)).contents
    ret = cam.MV_CC_CreateHandle(stDeviceList)
    if ret != 0 or cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0) != 0:
        print("Failed to open device!")
        sys.exit()

    cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
    stParam = MVCC_INTVALUE()
    memset(byref(stParam), 0, sizeof(MVCC_INTVALUE))
    ret = cam.MV_CC_GetIntValue("PayloadSize", stParam)
    nPayloadSize = stParam.nCurValue

    cam.MV_CC_StartGrabbing()

    # Создаем буфер для хранения данных кадра
    data_buf = (c_ubyte * nPayloadSize)()

    try:
        # Запускаем поток для захвата и обработки кадров
        hThreadHandle = threading.Thread(target=capture_and_process_video, args=(cam, data_buf))
        hThreadHandle.start()
        hThreadHandle.join()
    except Exception as e:
        print("Error:", e)

    # Остановка захвата и закрытие камеры
    cam.MV_CC_StopGrabbing()
    cam.MV_CC_CloseDevice()
    cam.MV_CC_DestroyHandle()

