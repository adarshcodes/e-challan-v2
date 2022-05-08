import cv2
import numpy as np

cap = cv2.VideoCapture('venv/lib/video.mp4')

min_width_react = 80
min_height_react = 80

count_line_position = 550
algo = cv2.bgsegm.createBackgroundSubtractorMOG()


def center_handle(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx, cy


detect = []
offset = 6

counter = 0
big_Vehicle = 0
truck = 0
two_wee = 0
car = 0

while True:
    ret, frame = cap.read()
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey, (3, 3), 5)
    img_sub = algo.apply(blur)

    dila = cv2.dilate(img_sub, np.ones((5, 5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dila2 = cv2.morphologyEx(dila, cv2.MORPH_CLOSE, kernel)
    dila2 = cv2.morphologyEx(dila2, cv2.MORPH_CLOSE, kernel)
    counterShape, h = cv2.findContours(dila2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.line(frame, (25, count_line_position), (1200, count_line_position), (255, 127, 0), 3)

    for (i, c) in enumerate(counterShape):
        (x, y, w, h) = cv2.boundingRect(c)
        validate_counter = (w >= min_width_react) and (h >= min_height_react)
        if not validate_counter:
            continue

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        if h+w >= 650:
            cv2.putText(frame, "Truck" + str(counter), (x+5, y - 50), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 1)
        elif h+w <= 172:
            cv2.putText(frame, "Two-Wheeler" + str(counter), (x + 5, y - 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 200, 120), 3)
        else:
            cv2.putText(frame, "Car" + str(counter), (x + 5, y - 50), cv2.FONT_HERSHEY_SIMPLEX, 3,
                        (255, 244, 70), 1)
        center = center_handle(x, y, w, h)
        detect.append(center)
        cv2.circle(frame, center, 4, (0, 0, 255), -1)

        for (x, y) in detect:
            if (count_line_position + offset) > y > (count_line_position - offset):
                counter += 1
            cv2.line(frame, (25, count_line_position), (1200, count_line_position), (0, 127, 255), 3)
            detect.remove((x, y))
            print("Red-Light Challan Breaker :" + str(w))

    cv2.putText(frame, "Red-Light Challan Breaker :" + str(counter), (50, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 55, 200), 2)
    cv2.imshow('Detector', dila2)
    cv2.imshow('Vehicle-Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
