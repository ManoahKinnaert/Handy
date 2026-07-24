import cv2
import pygame 
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import mediapipe.python.solutions.hands as mp_hands 


FEATURE_TABLE = ["WRIST", "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP", "INDEX_FINGER_MCP",
                 "INDEX_FINGER_PIP", "INDEX_FINGER_DIP", "INDEX_FINGER_TIP", "MIDDLE_FINGER_MCP",
                 "MIDDLE_FINGER_PIP", "MIDDLE_FINGER_DIP", "MIDDLE_FINGER_TIP", "RING_FINGER_MCP",
                 "RING_FINGER_PIP", "RING_FINGER_DIP", "RING_FINGER_TIP", "PINCKY_MCP", "PINCKY_PIP",
                 "PINCKY_DIP", "PINCKY_TIP"]

cube_vertices = ( 
    (1, -1, -1),
    (1, 1, -1),
    (-1, 1, -1),
    (-1, -1, -1),
    (1, -1, 1),
    (1, 1, 1),
    (-1, -1, 1),
    (-1, 1, 1)
)

cube_edges = (
    (0,1),
    (0,3),
    (0,4),
    (2,1),
    (2,3),
    (2,7),
    (6,3),
    (6,4),
    (6,7),
    (5,1),
    (5,4),
    (5,7)
)

# render the demo cube
def render_cube():
    glBegin(GL_LINES)
    for edge in cube_edges:
        for vertex in edge:
            glVertex3fv(cube_vertices[vertex])
    glEnd()


def delta(a: float, b: float):
    return b - a

def main():
    HEIGHT = 800
    WIDTH = HEIGHT * 16 // 9

    surface = pygame.display.set_mode((WIDTH, HEIGHT), DOUBLEBUF | OPENGL)
    pygame.display.set_caption("Cad Test#01")
    #running = True 
    fps_clock = pygame.time.Clock()
    fps = 24
    # init video capture
    cap = cv2.VideoCapture(index=1)

    # ensure scaling happens properly
    gluPerspective(45, (WIDTH / HEIGHT), 0.1, 50.0)
    glTranslatef(0, 0, -10)
    
    # keep track of the hands
    with mp_hands.Hands(
    model_complexity=0,
    max_num_hands=1,
    min_detection_confidence=.5,
    min_tracking_confidence=.5
    ) as hands:
        # actual mainloop
        tracked_features = [None for _ in FEATURE_TABLE]
        feature_positions = [None for _ in FEATURE_TABLE]
        while cap.isOpened():
            # handtracking
            success, frame = cap.read()
            if not success:
                print("[DEBUG]: Just ignoring empty camera frame.")
                continue
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
                            glTranslate(tracked_features[0][0], tracked_features[0][1], 0)
                            print(f"[DEBUG]: pos {FEATURE_TABLE[ids], cx, cy}") 
                            print(f"[DEBUG]: delta {FEATURE_TABLE[ids]} = {tracked_features[ids]}")
            
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            render_cube()
            pygame.display.flip()

            fps_clock.tick(fps)
            # handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    cap.release()
                    break
                
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        cap.release()
                        break 
        return
            
# main entry point
if __name__ == "__main__":
    main()
