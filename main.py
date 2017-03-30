# TODO: Spiking SOM

import numpy as np
import matplotlib.pyplot as plt
import cv2

cam = cv2.VideoCapture("lol.mp4")

previous_frame_dog = None
data_edges = np.array([])
data_movement = np.array([])

while True:
    (grabbed, frame) = cam.read()

    if not grabbed:
        break

    frame = cv2.resize(frame, dsize=(0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # frame_gray = cv2.GaussianBlur(frame_gray, (5, 5), 0)
    # frame_gray = cv2.dilate(frame_gray, None, iterations=2)

    g1 = cv2.GaussianBlur(frame_gray, (1, 1), 0)
    g2 = cv2.GaussianBlur(frame_gray, (3, 3), 0)
    frame_dog = g1 - g2

    amount_of_edges = cv2.countNonZero(frame_dog)
    data_edges = np.append(data_edges, amount_of_edges)

    frame_dog = cv2.threshold(frame_dog, 100, 255, cv2.THRESH_BINARY)[1]

    if previous_frame_dog is None:
        previous_frame_dog = frame_dog.copy()

    frame_temporal = cv2.subtract(frame_dog, previous_frame_dog)

    amount_of_movement = cv2.countNonZero(frame_temporal)
    data_movement = np.append(data_movement, amount_of_movement)

    cv2.imshow("SaturnV", frame_gray)
    cv2.imshow("DoG", frame_dog)
    cv2.imshow("Temporal", frame_temporal)

    previous_frame_dog = frame_dog.copy()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()

print(data_movement, data_movement.min(), data_movement.max())
print(data_edges, data_edges.min(), data_edges.max())

data = np.vstack([data_edges, data_movement])

plt.plot(range(0, len(data[0])), data[0], range(0, len(data[1])), data[1])

plt.show()


