import time

import mediapipe as mp
import cv2
import numpy as np


BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult

mp_hands = mp.tasks.vision.HandLandmarksConnections
mp_drawing = mp.tasks.vision.drawing_utils
mp_drawing_styles = mp.tasks.vision.drawing_styles

base_options=BaseOptions(model_asset_path="hand_landmarker.task")

class VideoHand:
    def cameraLoop(self):
        options = HandLandmarkerOptions(
            base_options=base_options,
            num_hands = 2,
            min_hand_detection_confidence = 0.8,
            min_hand_presence_confidence = 0.8,
            min_tracking_confidence = 0.8,
            # running_mode=VisionRunningMode.LIVE_STREAM,
            # result_callback=self.print_result,
        )
        
        cam = cv2.VideoCapture(index=0)
        
        with HandLandmarker.create_from_options(options) as detector:
            while(cam.isOpened):
                success, frame = cam.read()
                if not success:
                    print("No Frame!")
                    break
                
                frame = cv2.flip(frame, 1)
                h, w, _ = frame.shape
                
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                
                results = detector.detect(mp_image)
                
                # print(results.hand_landmarks, "\n")
                # print(results.hand_world_landmarks)
                
                if results.hand_world_landmarks:
                    for hand_landmarks in results.hand_world_landmarks:
                        index_tip = hand_landmarks[8]
                        print(index_tip)
                        x, y = int(index_tip.x* w) , int(index_tip.y * h)
                        cv2.circle(frame, (x, y), 50, (255, 0, 255) , -1)
                        
                cv2.imshow('MediaPipe Hands', frame)
            
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break  
            
        cam.release()
        cv2.destroyAllWindows()
        
        

if __name__ == "__main__":
    vh = VideoHand()
    vh.cameraLoop()
    # cameraLoop()
    # test()
