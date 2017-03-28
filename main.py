
# TODO: Preprocessing layer

import numpy as np
import cv2

cam = cv2.VideoCapture("SaturnV.mp4")

previous_frame_dog = None

while True:
    (grabbed, frame) = cam.read()

    if not grabbed:
        break

    frame = cv2.resize(frame, dsize=(0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    g1 = cv2.GaussianBlur(frame_gray, (1, 1), 0)
    g2 = cv2.GaussianBlur(frame_gray, (3, 3), 0)
    frame_dog = g1 - g2

    frame_dog = cv2.threshold(frame_dog, 100, 255, cv2.THRESH_BINARY)[1]

    if previous_frame_dog is None:
        previous_frame_dog = frame_dog.copy()

    frame_temporal = cv2.subtract(frame_dog, previous_frame_dog)

    cv2.imshow("SaturnV", frame_gray)
    cv2.imshow("DoG", frame_dog)
    cv2.imshow("Temporal", frame_temporal)

    previous_frame_dog = frame_dog.copy()

    if cv2.waitKey(40) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
