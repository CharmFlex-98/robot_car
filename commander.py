import pyautogui
import time
from utils import draw


def key_up():
    pyautogui.keyUp('a')
    pyautogui.keyUp('d')
    pyautogui.keyUp('w')


class commander:
    def __init__(self):
        pyautogui.PAUSE = 0
        self.midpoint_x = None
        self.midpoint_y = None
        self.has_target = False
        self.target_index = None
        self.present_key = None
        self.wander_mode = False
        self.frame_rate = 17
        self.counter = 0

    def find_position(self, class_to_be_tracking, sep_ratio, width, height):
        if not class_to_be_tracking:
            self.has_target = False
            print('no target!')
        else:
            self.has_target = False
            max_area = 0
            for index, item in enumerate(class_to_be_tracking):
                bbox_width = item[2]
                bbox_height = item[3]
                bbox_max_y = item[1] + bbox_height
                area = bbox_width * bbox_height

                if bbox_max_y <= height * (1 - sep_ratio):
                    self.has_target = True
                    print('has target!')
                    if max_area <= area:
                        max_area = area
                        self.target_index = index
                else:
                    continue
            if self.has_target:
                self.midpoint_x = int(class_to_be_tracking[self.target_index][0] +
                                      class_to_be_tracking[self.target_index][2] / 2)
                self.midpoint_y = int(class_to_be_tracking[self.target_index][1] +
                                      class_to_be_tracking[self.target_index][3] / 2)
            else:
                print('no target!')

    def action(self, sep_ratio, frame):
        width = frame.shape[1]
        height = frame.shape[0]
        if self.wander_mode:
            print(self.counter)
            pyautogui.keyDown('d')
            self.present_key = 'd'
            draw(frame, 'finding target...', 0, int(frame.shape[1] * 1 / 10))
            self.wander_mode = not self.wander_on(0.2)
            return

        if self.has_target:
            self.counter = 0
            left_lim = int(width / 2 - width * sep_ratio)
            right_lim = int(width / 2 + width * sep_ratio)
            if left_lim <= self.midpoint_x <= right_lim:
                if self.present_key == 'w':
                    pyautogui.keyDown('w')
                    print('forward!')
                else:
                    key_up()
                    pyautogui.keyDown('w')
                    self.present_key = 'w'
                    print('forward')
            elif self.midpoint_x <= left_lim:
                if self.present_key == 'a':
                    pyautogui.keyDown('a')
                    print('turn left!')
                else:
                    key_up()
                    pyautogui.keyDown('a')
                    self.present_key = 'a'
                    print('turn left!')
            elif self.midpoint_x >= right_lim:
                if self.present_key == 'd':
                    pyautogui.keyDown('d')
                    print('turn right!')
                else:
                    key_up()
                    pyautogui.keyDown('d')
                    self.present_key = 'd'
                    print('turn right!')
            else:
                print('error!')
        else:
            key_up()
            self.present_key = None
            self.wander_mode = self.wander_on(2)

    def wander_on(self, time_threshold):
        if self.counter > time_threshold * self.frame_rate:
            self.counter = 0
            return True
        else:
            self.counter += 1
            return False
