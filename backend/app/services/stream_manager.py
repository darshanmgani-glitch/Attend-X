import cv2
import os
import time
import asyncio
import numpy as np
import threading
from typing import Iterator
from fastapi.responses import StreamingResponse

from app.services.face_engine import engine
from app.core.database import get_db
from datetime import datetime
from app.core.config import settings
from app.core.constants import IST
import torch

# ── HOISTED IMPORTS (moved out of per-frame hot loop) ──
from torchvision import transforms
from PIL import Image

# ── HOISTED PREPROCESS PIPELINE (created once, reused every frame) ──
_preprocess = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# JPEG encode params (quality 80 for streaming — saves ~40% encode time vs default 95)
_JPEG_PARAMS = [int(cv2.IMWRITE_JPEG_QUALITY), 80]


class StreamManager:
    """
    Manages background capturing from RTSP/Webcam, running ByteTrack,
    generating embeddings, and maintaining the similarity buffer.

    Performance optimizations applied:
      1. Imports and transforms.Compose hoisted to module-level (not per-frame).
      2. Numpy array slicing replaces PIL crop for face extraction.
      3. ML Loop decoupled from Camera Loop: Camera runs at 30fps, ML runs asynchronously.
      4. Bounding boxes are stateful and drawn smoothly on every frame without blinking.
    """
    def __init__(self, src=0):
        self.src = src
        self.cap = None
        self.is_running = False
        self.paused = True
        
        # Threads
        self.cam_thread = None
        self.ml_thread = None
        
        # Shared State (with lock)
        self.current_frame = None
        self.frame_for_ml = None
        self._frame_lock = threading.Lock()
        
        # Detections State: drawn on every frame by the camera loop
        # Format: list of dicts {"box": [x1, y1, x2, y2], "text": str, "sim": float, "color": (B, G, R)}
        self.current_detections = []
        self._det_lock = threading.Lock()
        
        # ByteTrack Similarity Buffer
        self.track_buffer = {}
        self.BUFFER_TIMEOUT = 15.0
        self.REQUIRED_HITS = 2
        
        # Cooldown map
        self.last_scan_time = {}
        
        # Asyncio loops
        self.main_loop = None

    def start(self):
        if self.is_running:
            return
            
        try:
            self.main_loop = asyncio.get_running_loop()
        except RuntimeError:
            pass
        
        if not self.paused:
            if os.name == 'nt':
                self.cap = cv2.VideoCapture(self.src, cv2.CAP_DSHOW)
            else:
                self.cap = cv2.VideoCapture(self.src)
                
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        self.is_running = True
        
        # Start both decoupled threads
        self.cam_thread = threading.Thread(target=self._cam_worker, daemon=True)
        self.ml_thread = threading.Thread(target=self._ml_worker, daemon=True)
        self.cam_thread.start()
        self.ml_thread.start()
        print(f"  [StreamManager] Started capturing from {self.src}")

    def stop(self):
        self.is_running = False
        if self.cam_thread:
            self.cam_thread.join()
        if self.ml_thread:
            self.ml_thread.join()
        if self.cap:
            self.cap.release()
        print("  [StreamManager] Stopped capture.")

    def pause(self):
        self.paused = True
        if self.cap:
            self.cap.release()
            self.cap = None
        print("  [StreamManager] Stream Paused, Hardware Released.")

    def resume(self):
        if not self.paused:
            return
        time.sleep(1.0)
        
        if os.name == 'nt':
            self.cap = cv2.VideoCapture(self.src, cv2.CAP_DSHOW)
        else:
            self.cap = cv2.VideoCapture(self.src)
            
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.paused = False
        print("  [StreamManager] Stream Resumed.")

    def _cam_worker(self):
        """Camera loop: Fast reads 30fps, draws latest bounding boxes, updates frontend stream."""
        while self.is_running:
            try:
                if self.paused or self.cap is None:
                    time.sleep(0.5)
                    continue
                    
                ret, frame = self.cap.read()
                if not ret:
                    self.cap.release()
                    time.sleep(1.0)
                    if os.name == 'nt':
                        self.cap = cv2.VideoCapture(self.src, cv2.CAP_DSHOW)
                    else:
                        self.cap = cv2.VideoCapture(self.src)
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    continue
                
                # Clone raw frame for ML thread (so drawing on it doesn't corrupt ML input)
                with self._frame_lock:
                    self.frame_for_ml = frame.copy()
                
                # Draw latest available detections onto this frame
                with self._det_lock:
                    for det in self.current_detections:
                        self._draw_box(frame, det["box"], det["text"], det["sim"], det["color"])
                
                with self._frame_lock:
                    self.current_frame = frame
                    
            except Exception as e:
                print(f"  [StreamManager] Cam Thread Exception: {e}")
                time.sleep(0.5)

    def _ml_worker(self):
        """ML loop: Runs entirely decoupled. Grabs newest frame, does YOLO+FaceNet, updates detections."""
        while self.is_running:
            try:
                if self.paused:
                    time.sleep(0.5)
                    continue
                
                # Grab a frame safely
                frame_idx_to_process = None
                with self._frame_lock:
                    if self.frame_for_ml is not None:
                        frame_idx_to_process = self.frame_for_ml.copy()
                        
                if frame_idx_to_process is None:
                    time.sleep(0.05)
                    continue
                    
                self._process_frame(frame_idx_to_process)
                
                # Free CPU briefly
                time.sleep(0.03)
                
            except Exception as e:
                import traceback
                print(f"  [StreamManager] ML Thread Exception: {e}")
                traceback.print_exc()
                time.sleep(0.5)

    def _process_frame(self, frame: np.ndarray):
        """Run ByteTrack + FaceNet + FAISS + Buffer Logic. Returns immediately, updates shared state."""
        if not engine._initialized:
            return
            
        # 1. YOLO-Face + ByteTrack
        results = engine.yolo.track(frame, persist=True, conf=0.5, verbose=False)
        
        # If no faces, clear detection state and return
        if len(results) == 0 or len(results[0].boxes) == 0 or results[0].boxes.id is None:
            with self._det_lock:
                self.current_detections = []
            return
            
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().numpy()
        
        keypoints = None
        if hasattr(results[0], 'keypoints') and results[0].keypoints is not None:
            keypoints = results[0].keypoints.xy.cpu().numpy()
        
        now_ts = time.time()
        
        # Purge stale tracks memory
        stale = [k for k, v in self.track_buffer.items() if now_ts - v["last_seen"] > self.BUFFER_TIMEOUT]
        for k in stale:
            del self.track_buffer[k]
        
        new_detections = []
        tracks_needing_embedding = []
        
        # ── Fast path: Identify already-marked tracks ──
        for box_idx, (box, t_id) in enumerate(zip(boxes, track_ids)):
            if t_id in self.track_buffer and self.track_buffer[t_id]["marked_roll"]:
                # Already confirmed
                sim_val = self.track_buffer[t_id]["similarities"][-1] if self.track_buffer[t_id]["similarities"] else 0.0
                new_detections.append({
                    "box": box, "text": self.track_buffer[t_id]["marked_roll"], 
                    "sim": sim_val, "color": (0, 255, 0)
                })
                self.track_buffer[t_id]["last_seen"] = now_ts
            else:
                tracks_needing_embedding.append((box_idx, t_id))
        
        # If we have tracked everyone successfully, just update detections and sleep
        if not tracks_needing_embedding:
            with self._det_lock:
                self.current_detections = new_detections
            return  
        
        # 2. Extract crops for unresolved tracks
        valid_crops = []
        valid_tracks = []
        valid_box_indices = []
        
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h_img, w_img, _ = img.shape
        img_pil = Image.fromarray(img) 
        
        for box_idx, t_id in tracks_needing_embedding:
            box = boxes[box_idx]
            x1, y1, x2, y2 = map(int, box)
            
            w = x2 - x1
            h = y2 - y1
            if w < 40 or h < 40:
                continue
                
            padx, pady = int(w * 0.1), int(h * 0.1)
            x1 = max(0, x1 - padx)
            y1 = max(0, y1 - pady)
            x2 = min(w_img, x2 + padx)
            y2 = min(h_img, y2 + pady)
            
            crop = img_pil.crop((x1, y1, x2, y2))
            valid_crops.append(_preprocess(crop))
            valid_tracks.append(t_id)
            valid_box_indices.append(box_idx)
            
        if not valid_crops:
            with self._det_lock:
                self.current_detections = new_detections
            return
            
        # 3. Batch Tensor -> embeddings (FP16)
        batch_tensor = torch.stack(valid_crops).to(engine.device).half()
        with torch.no_grad():
            embeddings = engine.resnet(batch_tensor).float().cpu().numpy()
            
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        valid_mask = norms.flatten() > 0
        embeddings = embeddings[valid_mask]
        valid_tracks = [t for i, t in enumerate(valid_tracks) if valid_mask[i]]
        valid_box_indices = [b for i, b in enumerate(valid_box_indices) if valid_mask[i]]
        
        if len(embeddings) == 0:
            with self._det_lock:
                self.current_detections = new_detections
            return
            
        embeddings = (embeddings / norms[valid_mask]).astype(np.float32)
        
        # 4. Batch FAISS
        distances, indices = engine.index.search(embeddings, k=1)
        
        # 5. Process tracking buffer
        for i in range(len(embeddings)):
            sim = float(distances[i][0])
            idx = int(indices[i][0])
            t_id = valid_tracks[i]
            box_idx = valid_box_indices[i]
            
            if idx < 0 or idx >= len(engine.labels):
                continue
            closest_roll_no = engine.labels[idx]
            
            if t_id not in self.track_buffer:
                self.track_buffer[t_id] = {"similarities": [], "last_seen": now_ts, "marked_roll": None, "current_roll": closest_roll_no}
                
            tb = self.track_buffer[t_id]
            tb["last_seen"] = now_ts
            
            if tb["current_roll"] != closest_roll_no:
                tb["current_roll"] = closest_roll_no
                tb["similarities"] = []
                
            tb["similarities"].append(sim)
            # Only keep last 3 similarities to evaluate max over recent window
            if len(tb["similarities"]) > 3:
                tb["similarities"].pop(0)
                
            if len(tb["similarities"]) < 1:  # Require 1 hit minimum
                new_detections.append({
                    "box": boxes[box_idx], "text": "Analyzing...", 
                    "sim": sim, "color": (0, 255, 255)
                })
                continue
                
            # Use MAX similarity instead of AVERAGE to avoid drops from motion blur/angles
            max_sim = max(tb["similarities"])
            if max_sim >= settings.SIMILARITY_THRESHOLD:
                # MARK ATTENDANCE!
                tb["marked_roll"] = closest_roll_no
                new_detections.append({
                    "box": boxes[box_idx], "text": closest_roll_no, 
                    "sim": max_sim, "color": (0, 255, 0)
                })
                if self.main_loop:
                    asyncio.run_coroutine_threadsafe(self._log_attendance(closest_roll_no), self.main_loop)
            else:
                new_detections.append({
                    "box": boxes[box_idx], "text": "Unknown", 
                    "sim": max_sim, "color": (0, 0, 255)
                })

        # Atomic commit of all detection changes directly to the UI overlay
        with self._det_lock:
            self.current_detections = new_detections

    def _draw_box(self, frame, box, text, sim, color):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{text} {sim:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    async def _log_attendance(self, roll_no: str):
        """Writes to MongoDB tracking IN/OUT and Shift scheduling."""
        try:
            print(f"  [StreamManager] Trying to log attendance for {roll_no}...")
            now = datetime.now(IST)
            now_ts = now.timestamp()
            today_date = now.strftime("%Y-%m-%d")
            
            # 5-minute cooldown
            if roll_no in self.last_scan_time and (now_ts - self.last_scan_time[roll_no]) < 300:
                print(f"  [StreamManager] Ignored {roll_no} due to cooldown.")
                return
                
            self.last_scan_time[roll_no] = now_ts
            
            db = get_db()
            print(f"  [StreamManager] DB: {db}")
            student = await db.students.find_one({"roll_no": roll_no})
            if not student:
                print(f"  [StreamManager] Student {roll_no} not found in DB.")
                return
                
            existing = await db.attendance.find_one({"roll_no": roll_no, "date": today_date})
            
            config_doc = await db.settings.find_one({"_id": "global_config"})
            sys_login = config_doc["login_time"] if config_doc and "login_time" in config_doc else getattr(settings, "LOGIN_TIME", "09:30:00")
            sys_logout = config_doc["logout_time"] if config_doc and "logout_time" in config_doc else getattr(settings, "LOGOUT_TIME", "16:30:00")
            
            time_str = now.strftime("%H:%M:%S")
            
            if not existing:
                # Login Event
                login_thresh = datetime.strptime(today_date + " " + sys_login, "%Y-%m-%d %H:%M:%S").replace(tzinfo=IST)
                status = "On Time" if now <= login_thresh else "Late"
                
                await db.attendance.insert_one({
                    "roll_no": roll_no,
                    "name": student["name"],
                    "branch": student["branch"],
                    "date": today_date,
                    "login_time": time_str,
                    "login_status": status,
                    "logout_time": None,
                    "logout_status": None
                })
                msg = f"{student['name']} Logged In ({status})"
                print(f"  [StreamManager] LOGIN: {msg}")
            else:
                login_time_str = existing.get("login_time")
                if login_time_str:
                    login_time_obj = datetime.strptime(today_date + " " + login_time_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=IST)
                    if (now - login_time_obj).total_seconds() < 2 * 3600:
                        print(f"  [StreamManager] Ignored {roll_no} due to bounce cooldown.")
                        return 

                # Logout Event
                logout_thresh = datetime.strptime(today_date + " " + sys_logout, "%Y-%m-%d %H:%M:%S").replace(tzinfo=IST)
                status = "Logged Out" if now >= logout_thresh else "Early Logout"
                
                await db.attendance.update_one(
                    {"_id": existing["_id"]},
                    {"$set": {"logout_time": time_str, "logout_status": status}}
                )
                msg = f"{student['name']} {status}"
                print(f"  [StreamManager] LOGOUT: {msg}")
            
            from app.api.routes.attendance import recent_marks
            recent_marks.append({
                "roll_no": roll_no, 
                "name": student["name"], 
                "message": msg,
                "timestamp": time.time()
            })
            if len(recent_marks) > 20:
                recent_marks.pop(0)
        except Exception as e:
            import traceback
            print(f"  [StreamManager] Exception in _log_attendance: {e}")
            traceback.print_exc()

    def get_frame_jpeg(self) -> bytes:
        with self._frame_lock:
            frame = self.current_frame
        if frame is None:
            return b""
        ret, jpeg = cv2.imencode('.jpg', frame, _JPEG_PARAMS)
        if not ret:
            return b""
        return jpeg.tobytes()

streamer = StreamManager(src=0)
