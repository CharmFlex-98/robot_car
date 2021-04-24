import cv2


def draw(_frame, element, x_offset=0, y_offset=0):
    text_size = cv2.getTextSize(element, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
    x, y = int(0 + _frame.shape[1] / 20 + x_offset), int(text_size[1] + _frame.shape[0] / 20 + y_offset)
    cv2.putText(_frame, element, (x, y),
                cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
