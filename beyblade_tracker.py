import cv2 
import numpy as np
import pandas as pd
from ultralytics import YOLO
from scipy.spatial.distance import euclidean

class BeybladeBattleAnalyzer:
    def __init__(self, model_path, video_path, output_path):
        self.model = YOLO(model_path)
        self.video_path = video_path
        self.output_path = output_path

        # Needed Paramaters for stabilizing tracker ids
        self.stable_id_map = {} # Maps tracker_id to stable_id
        self.last_positions = {} # Last known positions for stable_id
        self.next_stable_id = 1 # Assume start with 1 beyblade first in the arena
        self.max_stable_ids = 2 # Only 2 beyblades, assume 1vs1

        # Battle Parameters to help extract features
        self.battle_started = False
        self.detected_broken_beyblade = False
        self.check_first_beyblade = True
        self.check_battle_ends = True
        self.flag_for_stopping_collision = True
        self.total_collision = 0

        # Helper Parameters to declare the winner
        # Beyblade 1 Definition is the first beyblade appear in the arena
        # Beyblade 2 Definition is the second beyblade appear in the arena
        self.prev_c1 = None # Previous Center Point of Beyblade 1
        self.prev_c2 = None # Preivous Center Point of Beyblade 2
        self.stop_frame_count_1 = 0
        self.stop_frame_count_2 = 0

    def midpoint(self, ptA, ptB):
        """
        To calculate the center point of a bounding box
        """
        return ((ptA[0] + ptB[0]) / 2, (ptA[1] + ptB[1]) / 2)

    def compute_iou(self, boxA, boxB):
        """
        To calculate the intersection over union
        box = (x1, y1, x2, y2)
        """
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
        return iou
    
    def assign_stable_id(self,track_id, x_center, y_center, stable_id_map, last_positions, next_stable_id, max_stable_ids):
        """
        To assign a stable id for each track id
        """
        if track_id not in stable_id_map:
            if next_stable_id <= max_stable_ids:
                stable_id_map[track_id] = next_stable_id
                next_stable_id += 1
            else:
                # Assign to closest stable ID based on distance
                dists = {
                    sid: np.linalg.norm(
                        np.array([x_center, y_center]) - np.array(last_positions.get(sid, [np.inf, np.inf]))
                    ) for sid in range(1, max_stable_ids + 1)
                }
                stable_id_map[track_id] = min(dists, key=dists.get)
        return stable_id_map[track_id], next_stable_id
    
    def check_first_beyblade_launch(self, stable_id, fps, cap):
        """
        To check whether the first beyblade has already been launch or not.
        """
        if self.check_first_beyblade and stable_id == 1:
            current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            current_time = current_frame / fps
            self.check_first_beyblade = False # Set to false, because only need one check

            # print("First Beyblade Launch!")
            # print(f"Frame: {current_frame}, Time: {current_time:.2f}s")

            return current_time
        
        return None
        
    def check_battle_start(self, last_positions, current_positions):
        """
        Checks if the battle started:
        - Both Beyblades (stable_id 1 and 2) detected
        - Both moving faster than speed_threshold (pixels/frame) # Limitations, not yet implemented
        """
        if 1 in last_positions and 2 in last_positions:
            return True
        
        elif 1 not in last_positions or 2 not in last_positions:
            return False  # Both not detected yet

        elif 1 not in current_positions or 2 not in current_positions:
            return False  # Both not detected in current frame
        
    def check_second_beyblade_and_battle(self, last_positions, current_positions, fps, cap):
        if not self.battle_started and self.check_battle_start(last_positions, current_positions):
            self.battle_started = True # Set to true, because only need one check
            current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            current_time = current_frame / fps

            # print("Second Beyblade Launch and Battle started!")
            # print(f"Frame: {current_frame}, Time: {current_time:.2f}s")
            return current_time
        
        return None
        
    def detect_collision(self, beyblade_1_positions, beyblade_2_positions, fps, cap):
        try:
            c1 = self.midpoint((beyblade_1_positions['x1'], beyblade_1_positions['y1']),
                               (beyblade_1_positions['x2'], beyblade_1_positions['y2']))
            c2 = self.midpoint((beyblade_2_positions['x1'], beyblade_2_positions['y1']),
                               (beyblade_2_positions['x2'], beyblade_2_positions['y2']))

            distance = euclidean(c1, c2)
            bbox_width = beyblade_2_positions['x2'] - beyblade_2_positions['x1']
            distance_thresh = bbox_width * 0.6

            boxA = list(map(int, beyblade_1_positions['box'].xyxy[0]))
            boxB = list(map(int, beyblade_2_positions['box'].xyxy[0]))
            iou = self.compute_iou(boxA, boxB)

            if (distance < distance_thresh or iou > 0.05) and self.flag_for_stopping_collision:
                current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                current_time = current_frame / fps
                self.total_collision += 1

                # print("💥Collision Happened!")
                # print(f"Frame: {current_frame}, Time: {current_time:.2f}s")
                
            self.prev_c1 = c1
            self.prev_c2 = c2

            return current_time, self.total_collision

        except:
            pass

    def check_battle_end(self, boxes, fps, cap):
        """
        This function only detect when the battle ends because the beyblade is broken into its parts.
        As for battle ends because one of the beyblade stop spinning still not yet implemented.
        Also, doesn't work if the one of the beyblade is out arena.
        """
        if len(boxes) > 2:
            self.detected_broken_beyblade = True

        if self.detected_broken_beyblade and self.check_battle_ends:
            current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            current_time = current_frame / fps
            
            self.check_battle_ends = False
            self.flag_for_stopping_collision = False
            
            # print("Battle ends!")
            # print(f"Frame: {current_frame}, Time: {current_time:.2f}s")

            return current_time, self.detected_broken_beyblade
        
        return None
    def declare_winner(self, c1, c2, prev_c1, prev_c2, detected_broken_beyblade):
        """
        This function only detect when the battle ends because the beyblade is broken into its parts.
        As for battle ends because one of the beyblade stop spinning still not yet implemented.
        Also, doesn't work if the one of the beyblade is out arena.
        Tie is also not yet implemented.
        """
        if detected_broken_beyblade:
            # Calculate velocity
            v1 = np.linalg.norm(np.array(c1) - np.array(prev_c1))
            v2 = np.linalg.norm(np.array(c2) - np.array(prev_c2))

            # Define stop threshold
            motion_thresh = 2  # pixels per frame
            self.stop_frame_count_1 += 1 if v1 < motion_thresh else 0
            self.stop_frame_count_2 += 1 if v2 < motion_thresh else 0

            # If one is stopped for N frames, declare the other the winner
            if self.stop_frame_count_1 > 30 and self.stop_frame_count_2 <= 30:
                print("🏆 Beyblade 2 wins!")
                return 'Beyblade 2'
            elif self.stop_frame_count_2 > 30 and self.stop_frame_count_1 <= 30:
                print("🏆 Beyblade 1 wins!")
                return ('Beyblade 1')

        return None

    def run_analysis(self):
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.output_path, fourcc, fps, (width, height))
    
        winner = ""
        type_of_finish = ""
        first_beyblade_launch_at = 0
        second_beyblade_launch_at = 0
        battle_start_at = 0
        first_collision_at = 0
        battle_duration = 0
        battle_end_time = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = self.model.track(
                frame,
                conf=0.5,
                tracker='botsort.yaml',
                persist=True,
                verbose=False
            )

            boxes = results[0].boxes
            current_positions = {}
            beyblade_1_positions = {}
            beyblade_2_positions = {}

            for box in boxes:
                if box.id is None:
                    continue
                track_id = int(box.id[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                x_center = (x1 + x2)/2
                y_center = (y1 + y2)/2
                
                # Assign Stable ID
                stable_id, self.next_stable_id = self.assign_stable_id(
                    track_id, x_center, y_center,
                    self.stable_id_map, self.last_positions,
                    self.next_stable_id, self.max_stable_ids
                )

                self.last_positions[stable_id] = (x_center, y_center)
              
                # To save the coordinates bbox for each object track per frame
                if stable_id == 1 and len(boxes) < 3:
                    beyblade_1_positions = {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'box': box}
                elif stable_id == 2 and len(boxes) < 3:
                    beyblade_2_positions = {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'box': box}

                # Draw box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Beyblade {stable_id}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

                # Launch First Beyblade Check
                launch_time = self.check_first_beyblade_launch(stable_id, fps, cap)
                if launch_time:
                    print(f"🚀 First Beyblade Launched at {launch_time:.2f}s")
                    first_beyblade_launch_at = launch_time


            # Second Beyblade launch & Battle start
            battle_time = self.check_second_beyblade_and_battle(self.last_positions, current_positions, fps, cap)
            if battle_time:
                print(f"⚔️ Battle Started at {battle_time:.2f}s")
                second_beyblade_launch_at = battle_time
                battle_start_at = battle_time

            # Collision
            if beyblade_1_positions and beyblade_2_positions:
                result = self.detect_collision(beyblade_1_positions, beyblade_2_positions, fps, cap)
                if result:
                    collision_time, total_collision = result
                    print(f"💥 Collision Detected at {collision_time:.2f}s")

                    first_collision_at = collision_time
            
            # Battle End
            end_result = self.check_battle_end(boxes, fps, cap)
            if end_result:
                end_time, self.detected_broken_beyblade = end_result
                print(f"🛑 Battle Ended at {end_time:.2f}s")
                battle_end_time = end_time

            # Winner
            if self.detected_broken_beyblade:
                try:
                    c1 = self.midpoint((beyblade_1_positions['x1'], beyblade_1_positions['y1']), 
                                       (beyblade_1_positions['x2'], beyblade_1_positions['y2']))
                    
                    c2 = self.midpoint((beyblade_2_positions['x1'], beyblade_2_positions['y1']),
                                       (beyblade_2_positions['x2'], beyblade_2_positions['y2']))
                    winner = self.declare_winner(c1, c2, self.prev_c1, self.prev_c2, self.detected_broken_beyblade)
                    self.prev_c1, self.prev_c2 = c1, c2
                    type_of_finish = 'broken beyblade'
                except:
                    pass

            battle_duration = battle_end_time - battle_start_at

            out.write(frame)
        summary = {
            "First Beyblade Launch Time (s)": [first_beyblade_launch_at],
            'Second Beyblade Launch Time (s)': [second_beyblade_launch_at],
            'Battle Start Time (s)': [battle_start_at],
            "First Collision Time (s)": [first_collision_at],
            "Total Collision": [collision_time],
            "Battle End Time (s)": [battle_end_time],
            "Winner": [winner],
            'Type of Finish': [type_of_finish],
            'Battle Duration': [battle_duration]
        }

        summary_df = pd.DataFrame(summary)
        summary_df.to_csv('battle_summary.csv',index=False)
        cap.release()    
        out.release()
        cv2.destroyAllWindows()
