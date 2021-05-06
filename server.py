import cv2
import random
import time
import zmq
import base64
import numpy as np
import commander
from utils import draw

# import gesture
# import pyautogui

cam_width, cam_height = 500, 500
cam = True
server = True
object_tracking = False
object_detection = True
gesture_control = False
target_item_name = 'person'
image_path = r'C:\Users\ASUS\PycharmProjects\Pytorch\Yolov3\cheongbakProject\01905.jpg'
names_path = '/home/jiaming/Desktop/darknet/darknet/data/coco.names'
cfg_path = '/home/jiaming/Desktop/darknet/darknet/cfg/yolov4-tiny.cfg'
weights_path = '/home/jiaming/Desktop/darknet/darknet/weights/yolov4-tiny.weights'
width = 416
height = 416
confidence_threshold = 0.2
nms_threshold = 0.4
class_colour_dict = {}


def find_object(_outputs, _frame):
    bboxes = []
    class_ids = []
    confidences = []
    frame_h, frame_w = _frame.shape[:2]
    for output in _outputs:  # each 3 different scales
        for detection in output:
            scores = detection[5:]
            detected_classID = scores.argmax(0)
            confidence = scores[detected_classID]
            if confidence > confidence_threshold:
                w, h = int(detection[2] * frame_w), int(detection[3] * frame_h)
                x, y = int(detection[0] * frame_w - w / 2), int(detection[1] * frame_h - h / 2)  # upper left coordinate
                bboxes.append([x, y, w, h])
                class_ids.append(detected_classID)
                confidences.append(float(confidence))

    object_index = cv2.dnn.NMSBoxes(bboxes, confidences, confidence_threshold, nms_threshold)
    for i in object_index:
        i = i[0]
        box = bboxes[i]
        x, y, w, h = box[0], box[1], box[2], box[3]

        collect_true_sample(class_names[class_ids[i]], [x, y, w, h])
        if class_names[class_ids[i]] == 'wdw':
            continue

        x1, y1, x2, y2 = bounding_limit(x, y, x + w, y + h, frame_h, frame_w)

        draw_class(_frame, class_names[class_ids[i]], confidences[i], x1, y1, x2, y2)


def collect_true_sample(class_name, coordinate):
    if class_name == 'we':
        person.append(coordinate)
    elif class_name == 'wewe':
        cell_phone.append(coordinate)
    elif class_name == target_item_name:
        target_item.append(coordinate)


def checking(_frame):
    frame_h = _frame.shape[0]
    frame_w = frame.shape[1]
    for p in person:
        equipped = False
        for index, item in enumerate(cell_phone):
            if p[0] <= item[0] + item[2] / 2 <= p[0] + p[2] and p[1] <= item[1] + item[3] / 2 <= p[1] + p[3]:
                x1, y1, x2, y2 = bounding_limit(p[0], p[1], p[0] + p[2], p[1] + p[3], frame_h, frame_w)
                draw_custom_class(_frame, 'good', x1, y1, x2, y2)
                equipped = True
                break
        if not equipped:
            x1, y1, x2, y2 = bounding_limit(p[0], p[1], p[0] + p[2], p[1] + p[3], frame_h, frame_w)
            draw_custom_class(_frame, 'bad', x1, y1, x2, y2)


def bounding_limit(x1, y1, x2, y2, max_h, max_w):
    x1 = min(max(x1, 0), max_w)
    x2 = min(max(x2, 0), max_w)
    y1 = min(max(y1, 0), max_h)
    y2 = min(max(y2, 0), max_h)

    return x1, y1, x2, y2


def draw_class(_frame, class_name, confidence, x1, y1, x2, y2):
    cv2.rectangle(_frame, (x1, y1), (x2, y2), class_colour_dict[class_name], 2)
    text_size = cv2.getTextSize('{} : {}%'.format(class_name, round(confidence * 100, 1)),
                                cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
    cv2.rectangle(_frame, (x1, y1), (x1 + text_size[0] + 8, y1 + text_size[1] + 8),
                  class_colour_dict[class_name], -1)

    cv2.putText(_frame, '{} : {}%'.format(class_name, round(confidence * 100, 1)),
                (x1 + 4, y1 + text_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [255, 255, 255], 1)


def draw_custom_class(_frame, custom_class, x1, y1, x2, y2):
    if custom_class == 'good':
        colour = [0, 255, 0]
    elif custom_class == 'bad':
        colour = [0, 0, 255]
    else:
        print('no such custom class!')
        return

    cv2.rectangle(_frame, (x1, y1), (x2, y2), colour, 2)
    text_size = cv2.getTextSize('{}!'.format(custom_class), cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
    cv2.rectangle(_frame, (x1, y1), (x1 + text_size[0] + 8, y1 + text_size[1] + 8),
                  colour, -1)
    cv2.putText(_frame, '{}!'.format(custom_class),
                (x1 + 4, y1 + text_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [255, 255, 255], 1)


def draw_target_class(_frame):
    text_size = cv2.getTextSize('{}'.format('TARGET'), cv2.FONT_HERSHEY_PLAIN, 1, 2)[0]
    x, y = int(command.midpoint_x - (text_size[0] / 2)), int(command.midpoint_y + (text_size[1] / 2))
    cv2.putText(_frame, '{}'.format('TARGET'),
                (x, y), cv2.FONT_HERSHEY_PLAIN, 1, [0, 0, 255], 2)


def class_colour(classes):
    for _class in classes:
        R, G, B = random.randint(0, 250), random.randint(0, 125), random.randint(0, 125)
        colour = (B, G, R)
        class_colour_dict[_class] = colour


def gesture_instruction(_frame):
    hand_gesture.preprocess(_frame)
    hand_gesture.find_hand(_frame)
    hand_gesture.find_position(_frame, [4, 8, 12], draw_bbox=False)

    finger_status = hand_gesture.finger_up()
    if finger_status[1:] == [1, 1, 1, 1]:
        pyautogui.keyDown('w')
    elif finger_status[1:] == [0, 1, 0, 0]:
        pyautogui.keyDown('s')
    elif finger_status[1:] == [0, 0, 0, 1]:
        pyautogui.keyDown('d')
    elif finger_status[1:] == [1, 0, 0, 0]:
        pyautogui.keyDown('a')
    else:
        commander.key_up()


with open(names_path, 'r') as file:
    class_names = file.read().rstrip('\n').split('\n')
    class_colour(class_names)

model = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
layers = model.getLayerNames()
outputNames = [layers[i[0] - 1] for i in model.getUnconnectedOutLayers()]

if gesture_control:
    server = True
    hand_gesture = gesture.HandDetector()
    my_cam = cv2.VideoCapture(0)

if cam:
    command = commander.commander()
    if not server:
        camera = cv2.VideoCapture(0, cv2.CAP_V4L2)
    else:
        """实例化用来接收帧的zmq对象"""
        context = zmq.Context()
        """zmq对象建立TCP链接"""
        footage_socket = context.socket(zmq.SUB)
        footage_socket.setsockopt_string(zmq.SUBSCRIBE, '')  # to receive as string
        footage_socket.setsockopt(zmq.CONFLATE, 1)  # only receive the last one in queue ( PAIR is not supported... )
        footage_socket.bind('tcp://192.168.0.15:5555')  # after initialize all the options, and put this last!!!

    while True:
        start_time = time.time()

        if gesture_control:
            cam_success, cam_frame = my_cam.read()
            gesture_instruction(cam_frame)

        if not server:
            success, frame = camera.read()
        else:
            frame = footage_socket.recv_string()  # 接收TCP传输过来的一帧视频图像数据
            if frame == 'tracking on':
                object_tracking = True
                continue
            elif frame == 'tracking off':
                object_tracking = False
                continue
            else:
                img = base64.b64decode(frame)  # 把数据进行base64解码后储存到内存img变量中
                npimg = np.frombuffer(img, dtype=np.uint8)  # 把这段缓存解码成一维数组
                frame = cv2.imdecode(npimg, 1)  # 将一维数组解码为图像source

        person = []
        cell_phone = []
        target_item = []

        if object_detection:
            blob = cv2.dnn.blobFromImage(frame, 1 / 255, (width, height), swapRB=True, crop=False)
            model.setInput(blob)
            outputs = model.forward(outputNames)  # these are the outputs with results

            find_object(outputs, frame)
            checking(frame)

            if object_tracking:
                command.find_position(target_item, 1 / 10, frame.shape[1], frame.shape[0])
                command.action(sep_ratio=1 / 6, frame=frame)
                if command.has_target:
                    draw_target_class(frame)

        fps = (round(1 / (time.time() - start_time)))
        draw(frame, 'FPS: {}'.format(fps))

        if gesture_control:
            frame = np.concatenate((frame, cam_frame), axis=1)

        cv2.imshow('cam', frame)
        cv2.waitKey(1)

else:
    image = cv2.imread(image_path)
    blob = cv2.dnn.blobFromImage(image, 1 / 255, (width, height), [0, 0, 0], crop=False)
    model.setInput(blob)

    layers = model.getLayerNames()
    outputNames = [layers[i[0] - 1] for i in model.getUnconnectedOutLayers()]
    outputs = model.forward(outputNames)  # these are the outputs with results

    person = []
    cell_phone = []

    find_object(outputs, image)
    checking(image)
    cv2.imwrite('opencv2.jpg', image)
