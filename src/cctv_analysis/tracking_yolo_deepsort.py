import cv2
import torch
from deep_sort_realtime.deepsort_tracker import DeepSort
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# ==== Step 1: Paths ====
base_dir = r"C:\Users\sharmbha\Downloads\Bhawna_project\Illegal_Immigration_AI_Project"
video_path = os.path.join(base_dir, "data", "processed", "cropped_footage.avi")
output_path = os.path.join(base_dir, "data", "final", "person_tracking_output.avi")  # Output to .avi to avoid potential MP4 issues

# ==== Step 2: Load YOLOv5 model ====
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', force_reload=False)
model.conf = 0.4
model.classes = [0]  # only detect persons

# ==== Step 3: Initialize DeepSORT ====
tracker = DeepSort(max_age=30)

# ==== Step 4: Load Video ====
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"[ERROR] Cannot open video: {video_path}")
    exit()

ret, frame = cap.read()
if not ret or frame is None:
    print("[ERROR] Couldn't read the first frame.")
    cap.release()
    exit()

frame_height, frame_width = frame.shape[:2]
print(f"[INFO] Video Resolution: {frame_width}x{frame_height}")

# ==== Step 5: Setup VideoWriter ====
fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # Use MJPG codec for better compatibility
out = cv2.VideoWriter(output_path, fourcc, 20.0, (frame_width, frame_height))

# Check if VideoWriter is initialized properly
if not out.isOpened():
    print("[ERROR] VideoWriter failed to open!")
    cap.release()
    exit()

# ==== Step 6: Process Each Frame ====
frame_id = 0
print("[INFO] Starting YOLO + DeepSORT tracking...")

while ret:
    frame_id += 1

    # Run YOLO detection on the current frame
    results = model(frame)
    detections = results.xyxy[0].cpu().numpy()

    person_detections = []
    for *box, conf, cls in detections:
        if cls == 0 and conf > 0.4:  # Check for person class and confidence threshold
            x1, y1, x2, y2 = map(int, box)
            person_detections.append(([x1, y1, x2 - x1, y2 - y1], conf, 'person'))

    # Debug: Check if any detections are found
    if len(person_detections) == 0:
        print(f"[INFO] No detections in frame {frame_id}.")

    # Update tracks with DeepSORT
    tracks = tracker.update_tracks(person_detections, frame=frame)

    # Draw the bounding boxes and track IDs
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        l, t, w, h = map(int, track.to_ltrb())
        r, b = l + w, t + h
        cv2.rectangle(frame, (l, t), (r, b), (0, 255, 0), 2)
        cv2.putText(frame, f'ID: {track_id}', (l, t - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Debugging: Show the frame before saving it (optional)
    # cv2.imshow("Frame", frame)
    # cv2.waitKey(1)

    # Save the frame as an image to check if the frame is being processed
    cv2.imwrite(os.path.join(base_dir, "debug", f"frame_{frame_id}.jpg"), frame)

    # Write frame to video if everything is good
    if out.isOpened():
        out.write(frame)
    else:
        print("[ERROR] Failed to write the frame.")
        break

    if frame_id % 10 == 0:  # Print every 10th frame for debugging
        print(f"[DEBUG] Frame {frame_id} - Persons Detected: {len(person_detections)}")

    # Read the next frame
    ret, frame = cap.read()
    if frame is None:
        print("[ERROR] Empty frame detected!")
        break

# ==== Step 7: Cleanup ====
cap.release()
out.release()
cv2.destroyAllWindows()
print(f"[DONE] Video saved to: {output_path}")
