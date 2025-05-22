# src/cctv_analysis/preprocess_video.py

import os
import cv2

def preprocess_video(input_video_path, output_video_path, crop_height=300):
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f" Could not open video: {input_video_path}")
        return

    frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f" Original Resolution: {frame_width}x{frame_height}, FPS: {fps}")

    new_height = min(crop_height, frame_height)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, new_height))

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cropped_frame = frame[:new_height, :]
        out.write(cropped_frame)
        frame_count += 1

    cap.release()
    out.release()
    print(f" Cropped Video saved at: {output_video_path}")
    print(f" Total Frames Processed: {frame_count}")


#  Call it here to test
input_path = r"C:\Users\sharmbha\Downloads\Bhawna_project\Illegal_Immigration_AI_Project\data\raw\real_border_footage.mp4"
output_path = r"C:\Users\sharmbha\Downloads\Bhawna_project\Illegal_Immigration_AI_Project\data\processed\cropped_footage.avi"

preprocess_video(input_path, output_path)
