# tracker.py
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from scipy.optimize import linear_sum_assignment
from reid import ReID

def calculate_iou(box1, box2):
    """Calculate Intersection over Union between two bounding boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0

class PlayerTracker:
    """
    Robust player tracker that combines motion-based and appearance-based tracking.
    """
    def __init__(self, iou_threshold=0.3, similarity_threshold=0.4, max_frames_to_forget=60):
        self.reid = ReID()
        self.tracked_players = {}
        self.next_player_id = 0
        
        # Thresholds
        self.iou_threshold = iou_threshold  # For frame-to-frame tracking
        self.similarity_threshold = similarity_threshold  # For re-identification
        self.max_frames_to_forget = max_frames_to_forget
        
        # For debugging
        self.frame_count = 0

    def _update_lost_tracks(self):
        """Remove tracks that have been lost for too long."""
        lost_ids = []
        for player_id, data in self.tracked_players.items():
            if data["frames_since_seen"] > self.max_frames_to_forget:
                lost_ids.append(player_id)
        for player_id in lost_ids:
            print(f"Frame {self.frame_count}: Removing Player {player_id} (lost for too long)")
            del self.tracked_players[player_id]

    def update(self, detections, frame):
        """
        Updates tracker with new detections, maintaining consistent IDs.
        """
        self.frame_count += 1
        
        # Update frames_since_seen for all tracks
        for player_id in self.tracked_players:
            self.tracked_players[player_id]["frames_since_seen"] += 1
        
        self._update_lost_tracks()
        
        if detections.size == 0:
            return []

        # Get embeddings for all detections
        new_embeddings = []
        valid_detections = []
        
        for bbox in detections:
            x1, y1, x2, y2 = map(int, bbox)
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)
            
            player_crop = frame[y1:y2, x1:x2]
            if player_crop.size == 0 or player_crop.shape[0] < 20 or player_crop.shape[1] < 20:
                continue
                
            try:
                embedding = self.reid.get_embedding(player_crop)
                new_embeddings.append(embedding)
                valid_detections.append(bbox)
            except:
                continue
        
        if not valid_detections:
            return []

        matched_tracks = []
        unmatched_det_indices = set(range(len(valid_detections)))
        unmatched_track_ids = set(self.tracked_players.keys())

        # STEP 1: Try to match using IOU (position-based) first
        if self.tracked_players:
            iou_matrix = np.zeros((len(valid_detections), len(self.tracked_players)))
            track_ids = list(self.tracked_players.keys())
            
            # Calculate IOU between all detections and existing tracks
            for i, det_bbox in enumerate(valid_detections):
                for j, track_id in enumerate(track_ids):
                    if self.tracked_players[track_id]["frames_since_seen"] <= 5:  # Only recent tracks
                        track_bbox = self.tracked_players[track_id]["bbox"]
                        iou_matrix[i, j] = calculate_iou(det_bbox, track_bbox)
            
            # Use Hungarian algorithm on IOU scores
            if iou_matrix.max() > self.iou_threshold:
                cost_matrix = 1 - iou_matrix
                row_ind, col_ind = linear_sum_assignment(cost_matrix)
                
                for r, c in zip(row_ind, col_ind):
                    if iou_matrix[r, c] >= self.iou_threshold:
                        track_id = track_ids[c]
                        
                        # Update the track
                        self.tracked_players[track_id]["bbox"] = valid_detections[r]
                        self.tracked_players[track_id]["frames_since_seen"] = 0
                        self.tracked_players[track_id]["hits"] += 1
                        
                        # Update embedding slowly
                        if new_embeddings:
                            self.tracked_players[track_id]["embedding"] = (
                                0.95 * self.tracked_players[track_id]["embedding"] + 
                                0.05 * new_embeddings[r]
                            )
                        
                        matched_tracks.append((valid_detections[r], track_id, 1.0))
                        unmatched_det_indices.discard(r)
                        unmatched_track_ids.discard(track_id)
                        
                        if self.frame_count < 10:  # Debug first few frames
                            print(f"Frame {self.frame_count}: Matched Player {track_id} by position (IOU={iou_matrix[r, c]:.2f})")

        # STEP 2: Try to match remaining detections using appearance (ReID)
        if unmatched_det_indices and unmatched_track_ids:
            # Get embeddings for unmatched items
            unmatched_det_list = list(unmatched_det_indices)
            unmatched_track_list = list(unmatched_track_ids)
            
            det_embeddings = np.array([new_embeddings[i] for i in unmatched_det_list])
            track_embeddings = np.array([self.tracked_players[tid]["embedding"] for tid in unmatched_track_list])
            
            # Calculate similarity
            sim_matrix = cosine_similarity(det_embeddings, track_embeddings)
            
            # Use Hungarian algorithm
            cost_matrix = 1 - sim_matrix
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            for r, c in zip(row_ind, col_ind):
                if sim_matrix[r, c] >= self.similarity_threshold:
                    det_idx = unmatched_det_list[r]
                    track_id = unmatched_track_list[c]
                    
                    # Update the track
                    self.tracked_players[track_id]["bbox"] = valid_detections[det_idx]
                    self.tracked_players[track_id]["frames_since_seen"] = 0
                    self.tracked_players[track_id]["hits"] += 1
                    self.tracked_players[track_id]["embedding"] = (
                        0.9 * self.tracked_players[track_id]["embedding"] + 
                        0.1 * new_embeddings[det_idx]
                    )
                    
                    matched_tracks.append((valid_detections[det_idx], track_id, sim_matrix[r, c]))
                    unmatched_det_indices.discard(det_idx)
                    
                    if self.frame_count < 10:
                        print(f"Frame {self.frame_count}: Re-identified Player {track_id} by appearance (sim={sim_matrix[r, c]:.2f})")

        # STEP 3: Create new tracks for remaining unmatched detections
        for det_idx in unmatched_det_indices:
            new_id = self.next_player_id
            self.tracked_players[new_id] = {
                "embedding": new_embeddings[det_idx],
                "bbox": valid_detections[det_idx],
                "frames_since_seen": 0,
                "hits": 1
            }
            matched_tracks.append((valid_detections[det_idx], new_id, 0))
            self.next_player_id += 1
            
            if self.frame_count < 10:
                print(f"Frame {self.frame_count}: Created new Player {new_id}")

        return matched_tracks