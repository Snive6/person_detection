from models.models import model, layer_name
from video_detection import detection_on_video
from image_detection import detection_image
import cv2
import os

imgs_directory = 'images'
for filename in os.scandir(imgs_directory):
    detection_image(cv2.imread(filename.path), model, layer_name)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

vds_directory = 'videos'
for filename in os.scandir(vds_directory):
    detection_on_video(cv2.VideoCapture(filename.path), model, layer_name)
