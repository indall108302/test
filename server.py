import cv2
import os
import asyncio
from fastapi import FastAPI, WebSocket, Request, Query
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from segmentation_engine import SegmentationEngine
from hik_camera import HikCamera
import uvicorn

app = FastAPI(title="YOLO Segmentation Dashboard API")

# Разрешаем CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = SegmentationEngine('runs/segment/train2/weights/best.pt')
hik_cam = HikCamera()

# Глобальные переменные для управления потоком
current_source = None # 'file' или 'camera'
current_video_path = None
is_running = False

@app.get("/api/videos")
async def list_videos():
    videos = [f for f in os.listdir(".") if f.endswith((".mp4", ".avi", ".mov"))]
    return {"videos": videos}

@app.get("/api/stats")
async def get_stats():
    return {
        "totals": engine.get_stats(),
        "class_names": engine.get_class_names()
    }

@app.post("/api/start")
async def start_processing(video_name: str):
    global current_video_path, is_running, current_source
    if os.path.exists(video_name):
        current_source = 'file'
        current_video_path = video_name
        is_running = True
        engine.reset()
        return {"status": "started", "video": video_name}
    return JSONResponse(status_code=404, content={"message": "Video not found"})

@app.post("/api/camera/start")
async def start_camera():
    global is_running, current_source
    success, msg = hik_cam.open()
    if success:
        if hik_cam.start():
            current_source = 'camera'
            is_running = True
            engine.reset()
            return {"status": "started", "source": "camera"}
        return JSONResponse(status_code=500, content={"message": "Не удалось запустить захват"})
    return JSONResponse(status_code=500, content={"message": msg})

@app.post("/api/stop")
async def stop_processing():
    global is_running
    is_running = False
    if current_source == 'camera':
        hik_cam.stop()
        hik_cam.close()
    return {"status": "stopped"}

def gen_frames():
    global is_running, current_video_path, current_source
    
    cap = None
    if current_source == 'file' and current_video_path:
        cap = cv2.VideoCapture(current_video_path)
    
    while is_running:
        frame = None
        if current_source == 'file' and cap:
            success, frame = cap.read()
            if not success:
                break
        elif current_source == 'camera':
            frame = hik_cam.get_frame()
            if frame is None:
                continue
        
        if frame is not None:
            # Обработка
            processed_frame = engine.process_frame(frame)
            
            # Кодирование в JPEG
            ret, buffer = cv2.imencode('.jpg', processed_frame)
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
    if cap:
        cap.release()
    if current_source == 'camera':
        hik_cam.stop()
        hik_cam.close()

@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(gen_frames(), media_type='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
