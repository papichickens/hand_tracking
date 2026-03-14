from time import sleep

import cv2

def cameraLoop():
    cam = cv2.VideoCapture(index=0)
    
    while(cam.isOpened):
        print("camera is up!")
        sleep(2)
    



if __name__ == "__main__":
    cameraLoop()