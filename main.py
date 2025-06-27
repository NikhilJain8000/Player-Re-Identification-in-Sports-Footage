# main.py
import cv2
import os
import argparse
from ultralytics import YOLO
from tqdm import tqdm
from tracker import PlayerTracker
import numpy as np

def get_player_color(player_id):
    """Returns a unique color for each player ID."""
    np.random.seed(player_id)  # Ensures same color for same ID
    color = np.random.randint(50, 255, size=3).tolist()
    return tuple(color)

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Loading YOLO detector...")
    detector = YOLO(args.yolo_model)
    print(f"Model classes: {detector.names}")
    
    print("Initializing Player Tracker...")
    tracker = PlayerTracker(iou_threshold=0.3, similarity_threshold=0.4)

    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {args.video_path}")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_path = os.path.join(args.output_dir, "tracked_output.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    print(f"Processing video: {args.video_path}")
    print("Tracking players, goalkeepers, and referees...")
    
    # Track statistics
    max_players_seen = 0
    
    for frame_idx in tqdm(range(total_frames), desc="Processing frames"):
        ret, frame = cap.read()
        if not ret:
            break

        # Detect players, goalkeepers, and referees (NOT the ball)
        results = detector(frame, classes=[1, 2, 3], verbose=False)
        
        if len(results[0].boxes) > 0:
            player_detections = results[0].boxes.xyxy.cpu().numpy()
        else:
            player_detections = np.array([])

        # Update tracker
        tracked_players = tracker.update(player_detections, frame)
        
        # Update statistics
        if len(tracked_players) > max_players_seen:
            max_players_seen = len(tracked_players)

        # Draw tracked players
        for bbox, player_id, score in tracked_players:
            x1, y1, x2, y2 = map(int, bbox)
            color = get_player_color(player_id)
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            
            # Create label
            label = f"ID:{player_id}"
            if score > 0 and score < 1:  # Show confidence for re-identified players
                label += f" ({score:.2f})"
            
            # Draw label background
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame, (x1, y1 - 30), (x1 + w + 10, y1), color, -1)
            cv2.putText(frame, label, (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Add info overlay
        info = f"Frame: {frame_idx} | Players: {len(tracked_players)} | Max: {max_players_seen}"
        cv2.rectangle(frame, (10, 10), (350, 40), (0, 0, 0), -1)
        cv2.putText(frame, info, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Draw the ball separately (optional)
        ball_results = detector(frame, classes=[0], verbose=False)
        if len(ball_results[0].boxes) > 0:
            ball_box = ball_results[0].boxes.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = map(int, ball_box)
            cv2.circle(frame, ((x1+x2)//2, (y1+y2)//2), 15, (0, 255, 255), 3)

        out.write(frame)

    print(f"\nProcessing complete!")
    print(f"Output saved to: {output_path}")
    print(f"Maximum players tracked simultaneously: {max_players_seen}")
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, default="15sec_input_720p.mp4")
    parser.add_argument("--yolo_model", type=str, default="models/player_ball_v11.pt")
    parser.add_argument("--output_dir", type=str, default="output")
    args = parser.parse_args()
    main(args)