# debug_yolo.py
import cv2
from ultralytics import YOLO

# Load the model
model = YOLO("models/player_ball_v11.pt")

# Get model info
print("Model Classes:", model.names)
print("Number of classes:", len(model.names))

# Test on a single frame
cap = cv2.VideoCapture("15sec_input_720p.mp4")
ret, frame = cap.read()

if ret:
    results = model(frame, verbose=False)
    
    print("\nDetections in first frame:")
    for box in results[0].boxes:
        class_id = int(box.cls)
        confidence = float(box.conf)
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        width = x2 - x1
        height = y2 - y1
        area = width * height
        
        print(f"Class: {model.names[class_id]}, Conf: {confidence:.2f}, Area: {int(area)}, Size: {int(width)}x{int(height)}")

cap.release()