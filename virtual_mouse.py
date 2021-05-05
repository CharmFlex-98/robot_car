import numpy as np

import hand_tracking
import cv2
import time
import autopy
import numpy

width, height = 500, 500
smoothing_value=3

camera = cv2.VideoCapture(0)
camera.set(3, width)
camera.set(4, height)
screen_w, screen_h = autopy.screen.size()

detector = hand_tracking.HandDetector(max_hand=1)
clicked = False
tempx, tempy=0, 0

while True:
    current_time = time.time()
    success, frame = camera.read()

    detector.preprocess(frame)
    detector.find_hand(frame)
    detector.find_position(frame, [8, 20])

    panelXmin, panelXmax = int(frame.shape[1] / 4), int(frame.shape[1] / 4 * 3)
    panelYmin, panelYmax = int(frame.shape[0] / 4), int(frame.shape[0] / 4 * 3)
    cv2.rectangle(frame, (panelXmin, panelYmin), (panelXmax, panelYmax), [255, 255, 255], 2)

    if detector.landmark_list:
        x1, y1 = detector.landmark_list[8][1:]

        finger_status = detector.finger_up()
        print(finger_status)

        if finger_status[2] == 0 and finger_status[3]==0 and finger_status[4]==0 and finger_status[1] == 1:
            clicked=False
            mouseX = np.interp(x1, (panelXmin, panelXmax), (0, screen_w))
            mouseX = max(min(mouseX, screen_w - 1), 1)
            mouseY = np.interp(y1, (panelYmin, panelYmax), (0, screen_h))
            mouseY = max(min(mouseY, screen_h - 1), 1)

            mouseX=tempx+(mouseX-tempx)/smoothing_value
            mouseY=tempy+(mouseY-tempy)/smoothing_value
            tempx = mouseX
            tempy = mouseY

            autopy.mouse.move(screen_w - mouseX, mouseY)

        if not clicked:
            if finger_status[4] == 1 and finger_status[1] == 1 and finger_status[2]==0 and finger_status[3]==0:
                autopy.mouse.click()
                clicked = True

    fps = round(1 / (time.time() - current_time), 2)
    cv2.putText(frame, 'FPS = {}'.format(fps), (20, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)

    cv2.imshow('cam', frame)
    cv2.waitKey(1)
