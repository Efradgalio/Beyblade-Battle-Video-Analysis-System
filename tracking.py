import os
import cv2
import time
import numpy as np

from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

import torch

vids = [1,2,3]
output_dir = './output_videos'
model = YOLO('./runs/detect/train5/weights/best.pt')  # Load the best model from training


# You initialize a DeepSort tracker that maintains object identities (IDs) across frames.
tracker = DeepSort(max_age=30, n_init=3, max_cosine_distance=0.4)

# max_age=30 means a tracked object can disappear for 30 frames before being removed.
# n_init=3 means an object must be detected for 3 consecutive frames before confirmed.
# max_cosine_distance=0.4 controls feature similarity threshold for matching detections to tracks.

# Parameters
speed_threshold = 2.0  # pixels/frame below which considered stopped
stop_frames_threshold = 10  # frames to confirm stop
arena_bounds = (0, 0, 640, 640)  # example arena size

# If a Beyblade moves less than 2 pixels between frames for 10 consecutive frames, it’s considered stopped.
# The arena is defined as a rectangle (top-left corner 0,0 to bottom-right 640,640).

# Tracking variables
stable_id_map = {}    # tracker_id -> stable_id (1 or 2)
last_positions = {}   # stable_id -> (x_center, y_center)
stopped_frames = {1: 0, 2: 0}  # count of consecutive frames below speed threshold
next_stable_id = 1

# Because tracker IDs can change due to occlusions or lost tracks, 
# you map them to stable IDs (1 or 2) representing Beyblade 1 and Beyblade 2.

# You keep track of each Beyblade’s last center position to calculate speed.
# Count frames with low movement to detect stopping.

cap = cv2.VideoCapture('./source video/beyblade battle 1 clean.mov')

# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Define codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or use 'XVID' for .avi
out = cv2.VideoWriter('./beyblade battle 1 clean.mov', fourcc, fps, (width, height))


while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    out.write(frame)

    # Run the YOLO model to get bounding boxes, confidences, and class IDs.
    # Convert boxes to format [x, y, width, height] expected by DeepSort.
    results = model(frame)[0]

    detections = []
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        detections.append(([x1, y1, x2 - x1, y2 - y1], conf, cls))


    # Update the tracker with the current frame's detections
    # Returns a list of tracked objects with their IDs and bounding boxes.
    tracks = tracker.update_tracks(detections, frame=frame)

    # Map tracker IDs to stable IDs
    for track in tracks:
        if not track.is_confirmed():
            continue

        tid = track.track_id
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)
        x_center = (x1 + x2) / 2
        y_center = (y1 + y2) / 2

        # Assign stable ID if new
        if tid not in stable_id_map:
            if next_stable_id <= 2:
                stable_id_map[tid] = next_stable_id
                next_stable_id += 1
            else:
                # Try to match this ID to closest last stable position
                dists = {sid: np.linalg.norm(np.array([x_center, y_center]) - np.array(last_positions.get(sid, [9999,9999]))) for sid in [1,2]}
                stable_id_map[tid] = min(dists, key=dists.get)

        stable_id = stable_id_map[tid]

        # Calculate speed (distance from last position)
        if stable_id in last_positions:
            dist = np.linalg.norm(np.array([x_center, y_center]) - np.array(last_positions[stable_id]))
        else:
            dist = 9999  # first frame, no speed

        last_positions[stable_id] = (x_center, y_center)

        # Check stopped or not
        if dist < speed_threshold:
            stopped_frames[stable_id] += 1
        else:
            stopped_frames[stable_id] = 0

        # Check if Beyblade exits arena
        out_of_bounds = (x1 < arena_bounds[0] or y1 < arena_bounds[1] or x2 > arena_bounds[2] or y2 > arena_bounds[3])

        # Draw box + stable ID
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(frame, f'Beyblade {stable_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        # If stopped or out of arena, declare winner
        if stopped_frames[stable_id] >= stop_frames_threshold or out_of_bounds:
            winner = 2 if stable_id == 1 else 1
            print(f"Winner is Beyblade {winner}")
            cap.release()
            out.release()
            cv2.destroyAllWindows()
            exit()

    cv2.imshow('Beyblade Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
