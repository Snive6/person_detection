from pedestrian_detection import pedestrian_detection
from models.models import LABELS
import cv2
import time
import imutils


def detection_image(image, model, layer_name):
	start = time.time()
	image = imutils.resize(image, width=700)
	results = pedestrian_detection(image, model, layer_name, personidz=LABELS.index("person"))
	c = 1
	for res in results:
		cv2.rectangle(image, (res[1][0], res[1][1]), (res[1][2], res[1][3]), (0, 255, 0), 2)
		cv2.putText(image, f'{c}', (res[1][0], res[1][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
		c += 1

	stop = time.time()
	timer = stop - start

	cv2.putText(image, f'Number of people counted: {c - 1} time: {round(timer, 5)}s', (20, 550), cv2.FONT_HERSHEY_DUPLEX,
				0.8, (0, 0, 255), 2)
	cv2.imshow("Detection", image)
