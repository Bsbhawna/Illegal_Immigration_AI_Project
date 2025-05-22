import cv2
import torch
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

# ==== 📁 Path Setup ====
base_dir = r"C:\Users\sharmbha\Downloads\Bhawna_project\Illegal_Immigration_AI_Project"

video_path = os.path.join(base_dir, r'data/processed/cropped_footage.avi')
output_video_path = os.path.join(base_dir, r'data/final/final_alert_overlay.avi')
log_csv_path = os.path.join(base_dir, r'data/logs/alert_log.csv')
timeline_chart_path = os.path.join(base_dir, r'data/final/alert_timeline.png')
summary_txt_path = os.path.join(base_dir, r'data/final/crowded_summary.txt')
sound_path = os.path.join(base_dir, r'data/assets/alert_sound.mp3')

# 📁 Ensure all folders exist
os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
os.makedirs(os.path.dirname(log_csv_path), exist_ok=True)
os.makedirs(os.path.dirname(timeline_chart_path), exist_ok=True)
os.makedirs(os.path.dirname(summary_txt_path), exist_ok=True)

# ==== 🚀 Load YOLOv5 Model ====
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', trust_repo=True)
model.eval()

# ==== 🎥 Load Video ====
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("❌ Error: Could not open video file.")
    exit()

frame_width, frame_height = int(cap.get(3)), int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"📽️ Total Frames: {total_frames}")

out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))

# ==== 📊 Init Logs ====
log_data = []
alert_count = 0
alert_threshold = 3
crowd_summary = []

from tqdm import tqdm
print("🚀 Starting video analysis...")
for frame_num in tqdm(range(total_frames)):
    ret, frame = cap.read()
    if not ret:
        break

    # 🔍 Run YOLO
    results = model(frame)
    df = results.pandas().xyxy[0]
    person_detections = df[df['name'] == 'person']
    num_persons = len(person_detections)

    alert = num_persons >= alert_threshold
    color = (0, 0, 255) if alert else (0, 255, 0)

    # 🧠 Draw info
    cv2.putText(frame, f"Persons: {num_persons}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    for _, row in person_detections.iterrows():
        x1, y1, x2, y2 = map(int, [row['xmin'], row['ymin'], row['xmax'], row['ymax']])
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    out.write(frame)

    # 🕒 Time from video frame
    timestamp = str(datetime.utcfromtimestamp(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000).strftime('%H:%M:%S'))

    log_data.append({
        'frame': frame_num,
        'timestamp': timestamp,
        'num_persons': num_persons,
        'alert': alert
    })

    if alert:
        alert_count += 1

    crowd_summary.append((frame_num, num_persons))

cap.release()
out.release()

# ==== 📝 Save Logs ====
log_df = pd.DataFrame(log_data)
log_df.to_csv(log_csv_path, index=False)

print(f"\n✅ Output video saved at: {output_video_path}")
print(f"📊 Total frames processed: {total_frames}")
print(f"🚨 Total alerts triggered: {alert_count}")
print(f"📝 Log saved to: {log_csv_path}")

# ==== 📈 Save Timeline Chart ====
plt.figure(figsize=(14, 5))
plt.plot(log_df['frame'], log_df['num_persons'], color='blue', label='Person Count')
plt.axhline(y=alert_threshold, color='red', linestyle='--', label='Alert Threshold')
plt.title("🚨 Alert Timeline")
plt.xlabel("Frame")
plt.ylabel("Detections")
plt.legend()
plt.tight_layout()
plt.savefig(timeline_chart_path)
plt.close()
print(f"📉 Timeline chart saved to: {timeline_chart_path}")

# ==== 🤖 Save Crowd Summary ====
crowd_summary.sort(key=lambda x: x[1], reverse=True)
top5 = crowd_summary[:5]

print("\n🤖 Top 5 Most Crowded Frames:")
for frame_id, count in top5:
    print(f"Frame {frame_id}: {count} persons")

with open(summary_txt_path, "w") as f:
    f.write("Top 5 Most Crowded Frames:\n")
    for frame_id, count in top5:
        f.write(f"Frame {frame_id}: {count} persons\n")
print(f"📄 Crowd summary saved to: {summary_txt_path}")

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import imageio
from datetime import datetime
import requests

# === 📂 Base Directory ===
base_dir = r"C:\Users\sharmbha\Downloads\Bhawna_project\Illegal_Immigration_AI_Project"
final_dir = os.path.join(base_dir, 'data/final')
os.makedirs(final_dir, exist_ok=True)

# === 🔢 Sample frame_alerts (replace with actual logic or loading) ===
frame_alerts = [1, 0, 3, 2, 0, 5, 0]  # Replace with real detection output
frame_count = len(frame_alerts)

# === 🔥 Heatmap Generation ===
heatmap_path = os.path.join(final_dir, 'heatmap_detections.png')
plt.figure(figsize=(10, 1))
sns.heatmap([frame_alerts], cmap='Reds', cbar=True, xticklabels=False, yticklabels=False)
plt.title('🚨 Illegal Crossing Heatmap')
plt.savefig(heatmap_path)
plt.close()
print(f"🔥 Heatmap saved at: {heatmap_path}")

# === 📊 Daily Summary CSV ===
dates = [datetime.now().strftime('%Y-%m-%d')] * frame_count
times = [f"{i:02d}:00" for i in range(frame_count)]
summary_df = pd.DataFrame({
    'date': dates,
    'time': times,
    'num_persons_detected': frame_alerts
})
daily_summary_path = os.path.join(final_dir, 'daily_summary.csv')
summary_df.to_csv(daily_summary_path, index=False)
print(f"📊 Daily Summary saved at: {daily_summary_path}")

# === 🎞️ GIF from alert frames (dummy visuals) ===
alert_frame_indices = [i for i, val in enumerate(frame_alerts) if val > 0]
frames = []
for idx in alert_frame_indices:
    frame_img = np.full((100, 300, 3), 255, dtype=np.uint8)
    cv2.putText(frame_img, f"Frame {idx} - Alerts: {frame_alerts[idx]}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    frames.append(frame_img)

if frames:
    gif_path = os.path.join(final_dir, 'alerts_summary.gif')
    imageio.mimsave(gif_path, frames, format='GIF', duration=0.8)
    print(f"🎞️ GIF saved at: {gif_path}")
else:
    print("⚠️ No alert frames found for GIF.")

# === 📤 Telegram Alert ===
def send_telegram_alert(token, chat_id, message, file_path=None):
    try:
        url_msg = f"https://api.telegram.org/bot{token}/sendMessage"
        data = {"chat_id": chat_id, "text": message}
        response_msg = requests.post(url_msg, data=data)

        if file_path and os.path.exists(file_path):
            url_file = f"https://api.telegram.org/bot{token}/sendDocument"
            with open(file_path, 'rb') as file:
                files = {"document": file}
                data_file = {"chat_id": chat_id}
                requests.post(url_file, data=data_file, files=files)
            print("📩 Telegram summary sent.")
        else:
            print(f"❌ Output video not found: {file_path}")
    except Exception as e:
        print("❌ Failed to send Telegram alert.", e)

# Fill these in with your credentials
TELEGRAM_TOKEN = "7658297256"
TELEGRAM_CHAT_ID = "1311"

output_video_path = os.path.join(base_dir, 'data/processed/final_alert_video.avi')

send_telegram_alert(TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, "🚨 Daily illegal immigration detection summary", output_video_path)
