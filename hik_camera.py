import numpy as np
import cv2
from MvImport.MvCameraControl_class import *

class HikCamera:
    def __init__(self):
        self.cam = MvCamera()
        self.is_opened = False
        self.is_grabbing = False
        self.data_buf = None
        self.payload_size = 0

    def open(self):
        """Поиск и открытие первой доступной камеры"""
        deviceList = MV_CC_DEVICE_INFO_LIST()
        tlayerType = MV_GIGE_DEVICE | MV_USB_DEVICE
        ret = MvCamera.MV_CC_EnumDevices(tlayerType, deviceList)
        
        if ret != 0 or deviceList.nDeviceNum == 0:
            return False, "Камера не найдена"

        stDeviceList = cast(deviceList.pDeviceInfo[0], POINTER(MV_CC_DEVICE_INFO)).contents
        ret = self.cam.MV_CC_CreateHandle(stDeviceList)
        if ret != 0:
            return False, f"Ошибка создания хендла: {ret}"

        ret = self.cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
        if ret != 0:
            return False, f"Ошибка открытия устройства: {ret}"

        # Настройки по умолчанию
        self.cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
        
        # Получаем размер данных
        stParam = MVCC_INTVALUE()
        self.cam.MV_CC_GetIntValue("PayloadSize", stParam)
        self.payload_size = stParam.nCurValue
        self.data_buf = (c_ubyte * self.payload_size)()
        
        self.is_opened = True
        return True, "Камера успешно открыта"

    def start(self):
        """Запуск захвата"""
        if not self.is_opened:
            return False
        ret = self.cam.MV_CC_StartGrabbing()
        if ret == 0:
            self.is_grabbing = True
            return True
        return False

    def get_frame(self):
        """Захват одного кадра"""
        if not self.is_grabbing:
            return None
        
        stFrameInfo = MV_FRAME_OUT_INFO_EX()
        ret = self.cam.MV_CC_GetOneFrameTimeout(self.data_buf, self.payload_size, stFrameInfo, 1000)
        
        if ret == 0:
            # Преобразование в numpy array
            img = np.asarray(self.data_buf).reshape(stFrameInfo.nHeight, stFrameInfo.nWidth, -1)
            # В Hikrobotics часто приходят кадры в Bayer или другом формате, 
            # но в примере пользователя используется простой резшейп и цветокоррекция
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) # Обычно Hik отдает RGB или Bayer
            return img
        return None

    def stop(self):
        """Остановка захвата"""
        if self.is_grabbing:
            self.cam.MV_CC_StopGrabbing()
            self.is_grabbing = False

    def close(self):
        """Закрытие камеры и освобождение ресурсов"""
        self.stop()
        if self.is_opened:
            self.cam.MV_CC_CloseDevice()
            self.cam.MV_CC_DestroyHandle()
            self.is_opened = False
