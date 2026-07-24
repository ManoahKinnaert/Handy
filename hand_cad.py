import cv2

import mediapipe.python.solutions.hands as mp_hands
import mediapipe.python.solutions.drawing_utils as mp_drawing
import mediapipe.python.solutions.drawing_styles as mp_drawing_styles
import asyncio

from ursina import *


FEATURE_TABLE = ["WRIST", "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP", "INDEX_FINGER_MCP",
                 "INDEX_FINGER_PIP", "INDEX_FINGER_DIP", "INDEX_FINGER_TIP", "MIDDLE_FINGER_MCP",
                 "MIDDLE_FINGER_PIP", "MIDDLE_FINGER_DIP", "MIDDLE_FINGER_TIP", "RING_FINGER_MCP",
                 "RING_FINGER_PIP", "RING_FINGER_DIP", "RING_FINGER_TIP", "PINCKY_MCP", "PINCKY_PIP",
                 "PINCKY_DIP", "PINCKY_TIP"]


async def move_model(model):
    print("test")
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

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Check the frame for hands
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)
            img_height, img_width, _ = frame_rgb.shape

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                        for ids, landmrk in enumerate(hand_landmarks.landmark):
                            cx, cy = landmrk.x, landmrk.y
                            if FEATURE_TABLE[ids] == "WRIST":
                                model.position = cx, 0, 0
                                print("test")
                
                for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                                image=frame,
                                landmark_list=hand_landmarks,
                                connections=mp_hands.HAND_CONNECTIONS,
                                landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
                                connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style(),
                            )


            # show the window and bind the letter q to terminating the program
            cv2.imshow("Hand Tracking", cv2.flip(frame, 1))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    cap.release()


def main():
    app = Ursina(borderless=False, title="Hand CAD")
    window.fps_counter.enabled = False 
    window.exit_button.visible = False

    cube = Entity(model="cube", color=color.red)
    cube.position = 1, 0, 0


    app.run()

if __name__ == "__main__":
    main()