import cv2

labelsPath = "models/coco.names"
LABELS = open(labelsPath).read().strip().split("\n")

weights_path = "models/yolov4-tiny.weights"
config_path = "models/yolov4-tiny.cfg"

model = cv2.dnn.readNetFromDarknet(config_path, weights_path)
# model = cv2.dnn.readNet(config_path, weights_path)
model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

layer_name = model.getLayerNames()
layer_name = [layer_name[i - 1] for i in model.getUnconnectedOutLayers()]

