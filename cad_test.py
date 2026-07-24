import cv2
import pygame 
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import mediapipe.python.solutions.hands as mp_hands 

from track import track_deltas
from utils.shapes import GlCube
from utils.constants import FEATURE_TABLE

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

    cube = GlCube()
    
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
            multi_hand_landmarks = track_deltas(cap, hands, tracked_features, feature_positions)
            if tracked_features[0] is not None and multi_hand_landmarks: glTranslate(tracked_features[0][0], tracked_features[0][1], 0)
           
            
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            cube.render()
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
