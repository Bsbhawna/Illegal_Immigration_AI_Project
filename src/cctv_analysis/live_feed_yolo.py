#  File: src/cctv_analysis/live_feed_yolo.py

import cv2
import torch
import pandas as pd
from datetime import datetime
import os
import winsound
import csv
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# === Paths and Setup ===
base_dir = r"C:\Users\sharmbha\Downloads\Bhawna_project\Illegal_Immigration_AI_Project"
#video_path = r'data/processed/cropped_footage.avi'
video_path = r"C:\Users\sharmbha\Downloads\Bhawna_project\Illegal_Immigration_AI_Project\data\processed\cropped_footage.avi"

output_video_path = r'data/final/final_alert_overlay.avi'
log_csv_path = r'data/logs/alert_log.csv'
sound_path = os.path.join(base_dir, r"data\assets\alert_sound.mp3")

# === Load YOLOv5 Model ===
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', force_reload=False)
model.classes = [0]  # Only detect persons

# === Load Video ===
if not os.path.exists(video_path):
    print(f"❌ File does not exist: {video_path}")
    exit()

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"❌ OpenCV could not open the video file. Check format or codec.")
    exit()

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
print(f"📽️ Total Frames: {total_frames}")

# === Setup Video Writer ===
out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'XVID'), 20.0, (frame_width, frame_height))

# === Prepare Log File ===
if not os.path.exists(log_csv_path):
    os.makedirs(os.path.dirname(log_csv_path), exist_ok=True)
    with open(log_csv_path, 'w', newline='') as file:
        csv.writer(file).writerow(["Timestamp", "Alert", "Frame", "Source"])

print("🚀 Starting video analysis...")

# === Analysis Loop ===
alert_count = 0
frame_count = 0

with tqdm(total=total_frames, desc="Processing Video Frames") as pbar:
    while frame_count < total_frames:
        ret, frame = cap.read()
        if not ret:
            print(f"⚠️ End of video or error reading frame {frame_count}.")
            break

        frame_count += 1
        pbar.update(1)

        results = model(frame)
        detections = results.pandas().xyxy[0]
        persons = detections[detections['name'] == 'person']

        if not persons.empty:
            alert_count += 1
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Draw bounding boxes
            for _, row in persons.iterrows():
                x1, y1, x2, y2 = map(int, [row['xmin'], row['ymin'], row['xmax'], row['ymax']])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, 'ALERT: Person Detected', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # Sound alert
            if os.path.exists(sound_path):
                try:
                    winsound.PlaySound(sound_path, winsound.SND_FILENAME | winsound.SND_ASYNC)
                except Exception as e:
                    print(f"🔇 Sound error: {e}")

            # Log alert
            with open(log_csv_path, 'a', newline='') as file:
                csv.writer(file).writerow([timestamp, "Person Detected", frame_count, "live_feed"])

        out.write(frame)

# === Cleanup ===
cap.release()
out.release()

print(f"\n✅ Output video saved at: {output_video_path}")
print(f"📊 Total frames processed: {frame_count}")
print(f"🚨 Total alerts triggered: {alert_count}")
print(f"📝 Log saved to: {log_csv_path}")
