import cv2

import mediapipe.python.solutions.hands as mp_hands
import mediapipe.python.solutions.drawing_utils as mp_drawing
import mediapipe.python.solutions.drawing_styles as mp_drawing_styles

from utils.math_helpers import delta 

FEATURE_TABLE = ["WRIST", "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP", "INDEX_FINGER_MCP",
                 "INDEX_FINGER_PIP", "INDEX_FINGER_DIP", "INDEX_FINGER_TIP", "MIDDLE_FINGER_MCP",
                 "MIDDLE_FINGER_PIP", "MIDDLE_FINGER_DIP", "MIDDLE_FINGER_TIP", "RING_FINGER_MCP",
                 "RING_FINGER_PIP", "RING_FINGER_DIP", "RING_FINGER_TIP", "PINCKY_MCP", "PINCKY_PIP",
                 "PINCKY_DIP", "PINCKY_TIP"]

def track_deltas(cap, hands, tracked_features, feature_positions):
    success, frame = cap.read()
    if not success:
         print("[DEBUG]: Just ignoring empty camera frame.")
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    img_h, img_w, _ = frame_rgb.shape
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for ids, landmrk in enumerate(hand_landmarks.landmark):
                if ids == 0:
                    cx, cy = landmrk.x * img_w, landmrk.y * img_h 
                    if tracked_features[0] is None and feature_positions[0] is None:
                        tracked_features[ids] = (cx, cy)
                        feature_positions[ids] = (cx, cy)
                    tracked_features[ids] = (-delta(feature_positions[ids][0], cx) / 1000, -delta(feature_positions[ids][1], cy) / 700)

                #print(f"[DEBUG]: pos {FEATURE_TABLE[ids], cx, cy}") 
                #print(f"[DEBUG]: delta {FEATURE_TABLE[ids]} = {tracked_features[ids]}")
    return results.multi_hand_landmarks
            
def run_hand_tracking_on_webcam():
    cap = cv2.VideoCapture(index=1)

    with mp_hands.Hands(
        model_complexity=0,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as hands:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Ignoring empty camera frame...")
                continue

            # Check the frame for hands
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)
            img_height, img_width, _ = frame_rgb.shape
            
            #print(get_hand_landmarks_pos(results, img_width, img_height))
            # display the landmarks
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                                image=frame,
                                landmark_list=hand_landmarks,
                                connections=mp_hands.HAND_CONNECTIONS,
                                landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
                                connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style(),)

            

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                        for ids, landmrk in enumerate(hand_landmarks.landmark):
                            cx, cy = landmrk.x * img_width, landmrk.y * img_height 
                            print(FEATURE_TABLE[ids], cx, cy)
        
            

            # show the window and bind the letter q to terminating the program
            cv2.imshow("Hand Tracking", cv2.flip(frame, 1))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()

# main entry point
if __name__ == "__main__":
    run_hand_tracking_on_webcam()
