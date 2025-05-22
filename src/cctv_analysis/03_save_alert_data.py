import cv2
import os
import csv
from datetime import timedelta

def save_alert_data(input_video_path, alert_csv_path, alert_trigger_fn=None):
    cap = cv2.VideoCapture(input_video_path)

    if not cap.isOpened():
        print(" Could not open video!")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    with open(alert_csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Frame', 'Timestamp', 'Alert_Triggered'])

        for frame_num in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break

            # Simulated alert condition (for now: every 30th frame triggers an alert)
            alert_triggered = False
            if alert_trigger_fn:
                alert_triggered = alert_trigger_fn(frame)
            else:
                alert_triggered = (frame_num % 30 == 0)

            timestamp = str(timedelta(seconds=frame_num / fps))
            writer.writerow([frame_num, timestamp, int(alert_triggered)])

    cap.release()
    print(f" Alerts saved to: {alert_csv_path}")
    
input_video = r"C:\Users\sharmbha\Downloads\Bhawna_project\Illegal_Immigration_AI_Project\data\final\final_alert_overlay.avi"
alert_csv = r"C:\Users\sharmbha\Downloads\Bhawna_project\Illegal_Immigration_AI_Project\data\final\alert_log.csv"

save_alert_data(input_video, alert_csv)
