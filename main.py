from time import sleep

import mediapipe as mp
import cv2
import numpy as np

def cameraLoop():
    cam = cv2.VideoCapture(index=0)
    
    while(cam.isOpened):
        print("camera is up!")
        sleep(2)
        
    
def test():
    
    ## setup available online
    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode
    
    mp_hands = mp.tasks.vision.HandLandmarksConnections
    mp_drawing = mp.tasks.vision.drawing_utils
    mp_drawing_styles = mp.tasks.vision.drawing_styles

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
        running_mode=VisionRunningMode.IMAGE,
    )
    
    with HandLandmarker.create_from_options(options) as detector:
        image = mp.Image.create_from_file("images/coisa.jpg")
        detection_result = detector.detect(image)

        # MediaPipe stores SRGB data here; OpenCV expects a NumPy array in BGR.
        image_bgr = cv2.cvtColor(image.numpy_view(), cv2.COLOR_RGB2BGR)
        print(f"Detected {len(detection_result.hand_landmarks)} hand(s)")
        # cv2.imshow("fds", image_bgr)
        # cv2.imshow("fds", cv2.flip(image_bgr, 1))
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        annotated_image = np.copy(image_bgr)
        mp_drawing.draw_landmarks(
            annotated_image,
            detection_result.hand_landmarks[0],
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style()
        )
        
        cv2.imshow("fds", annotated_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        

if __name__ == "__main__":
    # cameraLoop()
    test()
