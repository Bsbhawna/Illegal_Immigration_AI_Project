# src/cctv_analysis/ai_alert_overlay.py

import cv2
import os

def apply_alert_overlay(input_video_path, output_video_path):
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(" Could not open video!")
        return

    frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Simulated AI alert overlay
        cv2.rectangle(frame, (50, 50), (frame_width - 50, frame_height - 50), (0, 0, 255), 2)
        cv2.putText(frame, ' ALERT: Border Intrusion Detected',
                    (60, 80), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 255), 2, cv2.LINE_AA)

        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()
    print(f" Final Alert Video saved: {output_video_path}")
    print(f" Total Frames Processed: {frame_count}")
    
#  Call it here to test
input_video = r"C:\Users\sharmbha\Downloads\Bhawna_project\Illegal_Immigration_AI_Project\data\processed\cropped_footage.avi"
output_video = r"C:\Users\sharmbha\Downloads\Bhawna_project\Illegal_Immigration_AI_Project\data\final\final_alert_overlay.avi"

apply_alert_overlay(input_video, output_video)
