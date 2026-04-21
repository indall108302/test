import cv2
import os
import asyncio
from fastapi import FastAPI, WebSocket, Request, Query
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
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

# Монтируем статику фронтенда
app.mount("/assets", StaticFiles(directory="frontend/dist/assets"), name="assets")

@app.get("/")
async def serve_index():
    return FileResponse("frontend/dist/index.html")

@app.middleware("http")
async def log_requests(request: Request, call_next):
    print(f"DEBUG: Incoming {request.method} request to {request.url.path}")
    response = await call_next(request)
    return response

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

@app.get("/api/start")
async def start_processing(video_name: str):
    print(f"DEBUG: START command received (GET) for: {video_name}")
    global current_video_path, is_running, current_source
    # Очищаем старое состояние
    is_running = False
    await asyncio.sleep(0.1) 
    
    if os.path.exists(video_name):
        current_source = 'file'
        current_video_path = video_name
        is_running = True
        engine.reset()
        print(f"DEBUG: Set source to {current_source} with path {current_video_path}")
        return {"status": "started", "video": video_name}
    else:
        print(f"ERROR: File {video_name} not found in {os.getcwd()}")
    return JSONResponse(status_code=404, content={"message": "Video not found"})

@app.get("/api/camera/start")
async def start_camera():
    global is_running, current_source
    print("DEBUG: Camera START command received (GET)")
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
    
    print(f"DEBUG: Starting stream. Source: {current_source}, Path: {current_video_path}")
    
    cap = None
    if current_source == 'file' and current_video_path:
        cap = cv2.VideoCapture(current_video_path)
        if not cap.isOpened():
            print(f"ERROR: Could not open video file {current_video_path}")
            is_running = False
            return

    frame_count = 0
    while is_running:
        frame = None
        if current_source == 'file' and cap:
            success, frame = cap.read()
            if not success:
                print("DEBUG: End of video file or read error")
                break
        elif current_source == 'camera':
            frame = hik_cam.get_frame()
            if frame is None:
                continue
        
        if frame is not None:
            frame_count += 1
            if frame_count % 30 == 0:
                print(f"DEBUG: Processed {frame_count} frames...")
                
            try:
                # Обработка
                processed_frame = engine.process_frame(frame)
                
                # Кодирование в JPEG
                ret, buffer = cv2.imencode('.jpg', processed_frame)
                if not ret:
                    print("ERROR: Failed to encode frame to JPEG")
                    continue
                
                frame_bytes = buffer.tobytes()
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            except Exception as e:
                print(f"ERROR during processing: {e}")
                break
        
    print("DEBUG: Stream stopped")
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
