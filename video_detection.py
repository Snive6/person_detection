from pedestrian_detection import pedestrian_detection
from models.models import LABELS
import cv2
import time
import imutils


def detection_on_video(cap, model, layer_name):
    while True:
        start = time.time()
        (grabbed, image) = cap.read()

        if not grabbed:
            break
        # image = cv2.resize(image, [800, 600])
        image = imutils.resize(image, width=700)
        results = pedestrian_detection(image, model, layer_name, personidz=LABELS.index("person"))

        c = 1
        for res in results:
            cv2.rectangle(image, (res[1][0], res[1][1]), (res[1][2], res[1][3]), (0, 255, 0), 2)
            cv2.putText(image, f'{c}', (res[1][0], res[1][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            c += 1
        cv2.putText(image, f'Liczba zliczonych osob: {c - 1}', (20, 550), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255),
                    2)
        cv2.imshow("Detection", image)
        cv2.waitKey(1)
        stop = time.time()
        print(round(stop - start, 4), 's')

        if cv2.getWindowProperty('Detection', cv2.WND_PROP_VISIBLE) < 1:
            break
    cv2.destroyAllWindows()
