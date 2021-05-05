import cv2
import mediapipe as mp
import time


class HandDetector():
    def __init__(self, mode=False, max_hand=2, detection_conf_thres=0.5, tracking_conf_thres=0.5):
        self.mode = mode
        self.max_hand = max_hand
        self.detection_threshold = detection_conf_thres
        self.tracking_threshold = tracking_conf_thres
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands()
        self.mpDraw = mp.solutions.drawing_utils
        self.results = ''
        self.landmark_list=[]

    def preprocess(self, frame):
        imageRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imageRGB)

    def find_hand(self, _frame, draw=True):
        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks[:self.max_hand]:  # for each hand
                if draw:
                    self.mpDraw.draw_landmarks(_frame, hand_landmarks, self.mpHands.HAND_CONNECTIONS)

    def find_position(self, _frame, hand_id, hand_ID=0, draw=True, draw_bbox=True, bbox_shiftx=10, bbox_shifty=10):
        self.landmark_list=[]
        min_x, min_y, max_x, max_y = 10000, 10000, 0, 0
        if self.results.multi_hand_landmarks:
            target_hand = self.results.multi_hand_landmarks[hand_ID]
            for id, landmark in enumerate(target_hand.landmark):
                h, w, c = _frame.shape
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                self.landmark_list.append([id, cx, cy])
                if id in hand_id:
                    cv2.circle(_frame, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

                if cx > max_x:
                    max_x = cx
                if cy > max_y:
                    max_y = cy
                if cx < min_x:
                    min_x=cx
                if cy < min_y:
                    min_y=cy

            if draw_bbox:
                bbox_minx=min_x-bbox_shiftx
                bbox_miny=min_y-bbox_shifty
                bbox_maxx=max_x+bbox_shiftx
                bbox_maxy=max_y+bbox_shifty
                cv2.rectangle(_frame, (bbox_minx, bbox_miny), (bbox_maxx, bbox_maxy), [255, 255, 255], 2)

    def finger_up(self):
        finger_status=[]
        if self.landmark_list:
            for id in [4, 8, 12, 16, 20]:
                if self.landmark_list[id][2]<self.landmark_list[id-1][2]:
                    finger_status.append(1)
                else:
                    finger_status.append(0)

        return finger_status




if __name__ == '__main__':
    hand_detector = HandDetector()
    camera = cv2.VideoCapture(0)

    while True:
        current_time = time.time()

        success, frame = camera.read()

        hand_detector.preprocess(frame)
        hand_detector.find_hand(frame, draw=False)
        hand_detector.find_position(frame, 0)

        fps = round(1 / (time.time() - current_time), 2)
        cv2.putText(frame, 'FPS = {}'.format(fps), (20, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)

        cv2.imshow('cam', frame)
        cv2.waitKey(1)
