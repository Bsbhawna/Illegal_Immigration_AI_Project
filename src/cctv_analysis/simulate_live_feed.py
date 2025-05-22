import cv2
import random
import time

# Path to your video file
video_path = "C:/Users/sharmbha/Downloads/Bhawna_project/Illegal_Immigration_AI_Project/data/processed/cropped_footage.avi"

# Open the video file
cap = cv2.VideoCapture(video_path)

# Check if video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

while True:
    # Read the next frame from the video
    ret, frame = cap.read()
    
    if not ret:
        # If we reach the end of the video, restart it
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue
    
    # Simulate detection by randomly drawing bounding boxes
    # Randomly select coordinates for the bounding box
    height, width, _ = frame.shape
    x1 = random.randint(0, width - 100)
    y1 = random.randint(0, height - 100)
    x2 = x1 + random.randint(50, 150)
    y2 = y1 + random.randint(50, 150)
    
    # Draw the bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Add a label/text inside the box (simulating detection)
    label = "Crossing Detected"
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, label, (x1, y1 - 10), font, 0.8, (0, 255, 0), 2)

    # Add timestamp to the frame (simulating real-time)
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    cv2.putText(frame, timestamp, (10, height - 10), font, 0.7, (255, 255, 255), 2)
    
    # Show the frame (Live feed simulation)
    cv2.imshow("Live Feed Simulation", frame)

    # Wait for a key press to continue
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break  # Press 'q' to quit

# Release the video capture object and close the window
cap.release()
cv2.destroyAllWindows()
