import cv2
import pygame 
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import mediapipe.python.solutions.hands as mp_hands 

from tests.experimental.track import track_deltas
from utils.shapes import GlCube
from utils.constants import FEATURE_TABLE


HEIGHT = 800
WIDTH = HEIGHT * 16 // 9

DRAG_ENABLED = False 
TRACKED_FEATURES = [None for _ in FEATURE_TABLE]
FEATURE_POSITIONS = [None for _ in FEATURE_TABLE]

def render(objects):
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    for obj in objects:
        obj.render()
    pygame.display.flip() 

def handle_events(cap):
    global DRAG_ENABLED
    global TRACKED_FEATURES, FEATURE_POSITIONS

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            cap.release()
            break
                
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_d:
                DRAG_ENABLED = not DRAG_ENABLED 
                print("[DEBUG]: D pressed", DRAG_ENABLED)
            
            # choose a new reference point for the hand
            elif event.key == pygame.K_c:
                TRACKED_FEATURES = [None for _ in FEATURE_TABLE]
                FEATURE_POSITIONS = [None for _ in FEATURE_TABLE] 
            
            elif event.key == pygame.K_q:
                cap.release()
                break 
            
def main():
    global DRAG_ENABLED
    surface = pygame.display.set_mode((WIDTH, HEIGHT), DOUBLEBUF | OPENGL)
    pygame.display.set_caption("Cad Test#01")
    fps_clock = pygame.time.Clock()
    fps = 24
    # init video capture
    cap = cv2.VideoCapture(0)

    # ensure scaling happens properly
    gluPerspective(45, (WIDTH / HEIGHT), 0.1, 50.0)
    glTranslatef(0, 0, -10)

    objects = [GlCube()]
    
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
            # handle events
            handle_events(cap)
            multi_hand_landmarks = track_deltas(cap, hands, tracked_features, feature_positions)
            if tracked_features[0] is not None and multi_hand_landmarks and DRAG_ENABLED: glTranslate(tracked_features[0][0], tracked_features[0][1], 0)
            
            render(objects)
            fps_clock.tick(fps)
           
        return
            
# main entry point
if __name__ == "__main__":
    main()
