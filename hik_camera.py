import numpy as np
import cv2
import asyncio
from ctypes import *

try:
    from MvImport.MvCameraControl_class import *
    SDK_AVAILABLE = True
except ImportError:
    SDK_AVAILABLE = False
    # Заглушки для типов данных (для запуска без SDK)
    class MvCamera: pass
    def cast(obj, type): return obj
    def POINTER(type): return type
    def byref(obj): return obj
    def sizeof(obj): return 0
    def memset(a, b, c): pass
    c_ubyte = None
    MV_GIGE_DEVICE = 0
    MV_USB_DEVICE = 0
    MV_ACCESS_Exclusive = 0
    MV_TRIGGER_MODE_OFF = 0
    class MV_CC_DEVICE_INFO_LIST:
        def __init__(self): self.nDeviceNum = 0
    class MV_FRAME_OUT_INFO_EX: pass
    class MVCC_INTVALUE: 
        def __init__(self): self.nCurValue = 0
    class MV_CC_DEVICE_INFO: pass

class HikCamera:
    def __init__(self):
        self.cam = MvCamera()
        self.is_opened = False
        self.is_grabbing = False
        self.data_buf = None
        self.payload_size = 0

    def open(self):
        if not SDK_AVAILABLE:
            return False, "SDK MvImport не найден в системе"
            
        deviceList = MV_CC_DEVICE_INFO_LIST()
        tlayerType = MV_GIGE_DEVICE | MV_USB_DEVICE
        ret = MvCamera.MV_CC_EnumDevices(tlayerType, deviceList)
        
        if ret != 0 or deviceList.nDeviceNum == 0:
            return False, "Камера Hikrobotics не обнаружена. Проверьте подключение и питание."

        stDeviceList = cast(deviceList.pDeviceInfo[0], POINTER(MV_CC_DEVICE_INFO)).contents
        ret = self.cam.MV_CC_CreateHandle(stDeviceList)
        if ret != 0:
            return False, f"Ошибка создания хендла: {ret}"

        ret = self.cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
        if ret != 0:
            return False, f"Не удалось открыть устройство: {ret}"

        self.cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
        
        stParam = MVCC_INTVALUE()
        memset(byref(stParam), 0, sizeof(MVCC_INTVALUE))
        self.cam.MV_CC_GetIntValue("PayloadSize", stParam)
        self.payload_size = stParam.nCurValue
        self.data_buf = (c_ubyte * self.payload_size)()
        
        self.is_opened = True
        return True, "Камера успешно инициализирована"

    def start(self):
        if not self.is_opened: return False
        ret = self.cam.MV_CC_StartGrabbing()
        if ret == 0:
            self.is_grabbing = True
            return True
        return False

    def get_frame(self):
        if not self.is_grabbing: return None
        
        stFrameInfo = MV_FRAME_OUT_INFO_EX()
        memset(byref(stFrameInfo), 0, sizeof(stFrameInfo))
        ret = self.cam.MV_CC_GetOneFrameTimeout(self.data_buf, self.payload_size, stFrameInfo, 1000)
        
        if ret == 0:
            # Логика в точности как в вашем Detect_and_cap.py
            img = np.asarray(self.data_buf).reshape(stFrameInfo.nHeight, stFrameInfo.nWidth, -1)
            img = cv2.resize(img, (1920, 1080))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Соответствует вашему примеру
            return img
        return None

    def stop(self):
        if self.is_grabbing:
            self.cam.MV_CC_StopGrabbing()
            self.is_grabbing = False

    def close(self):
        self.stop()
        if self.is_opened:
            self.cam.MV_CC_CloseDevice()
            self.cam.MV_CC_DestroyHandle()
            self.is_opened = False
