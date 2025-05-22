import cv2
import torch
import time
import os
import csv
from datetime import datetime
import winsound

# Paths
input_video = r"C:\Users\sharmbha\Downloads\Bhawna_project\Illegal_Immigration_AI_Project\data\processed\cropped_footage.avi"
output_video_path = r"C:\Users\sharmbha\Downloads\Bhawna_project\Illegal_Immigration_AI_Project\data\final\final_alert_overlay.avi"
log_csv = r"C:\Users\sharmbha\Downloads\Bhawna_project\Illegal_Immigration_AI_Project\data\logs\alert_log.csv"
sound_path = r'C:\Users\sharmbha\Downloads\Bhawna_project\Illegal_Immigration_AI_Project\data\assets\alert_sound.mp3'

# Load YOLOv5 Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.conf = 0.4  # Confidence threshold

# Open video
cap = cv2.VideoCapture(input_video)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# Logging setup
os.makedirs(os.path.dirname(log_csv), exist_ok=True)

# Initialize log file with headers (overwrite previous file if needed)
with open(log_csv, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Timestamp", "Alert", "Frame", "Source"])

frame_count = 0
alert_count = 0

print(" Starting video analysis...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    results = model(frame)
    detections = results.pandas().xyxy[0]
    persons = detections[detections['name'] == 'person']

    if len(persons) > 0:
        alert_count += 1
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        for _, row in persons.iterrows():
            x1, y1, x2, y2 = map(int, [row['xmin'], row['ymin'], row['xmax'], row['ymax']])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, "ALERT", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Log alert
        with open(log_csv, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([timestamp, "Person Detected", frame_count, "yolo_alert_overlay"])

        # Optional sound
        if os.path.exists(sound_path):
            try:
                winsound.PlaySound(sound_path, winsound.SND_FILENAME | winsound.SND_ASYNC)
            except Exception as e:
                print("Sound error:", e)

    out.write(frame)

# Cleanup
cap.release()
out.release()
try:
    cv2.destroyAllWindows()
except:
    pass

print(f"\n YOLO Alert Video saved: {output_video_path}")
print(f" Total Frames: {frame_count},  Alerts Triggered: {alert_count}")
print(f" Log saved to: {log_csv}")
