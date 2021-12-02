from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import time
import os
import cv2 as cv

base_dir = os.path.dirname(__file__)
threshold = 0.5  # CONFIDENCE THRESHOLD

# LOAD MODEL
prototxt_file = '01_model/Resnet_SSD_deploy.prototxt'
caffemodel_file = '01_model/Res10_300x300_SSD_iter_140000.caffemodel'
net = cv.dnn.readNetFromCaffe(prototxt_file, caffeModel=caffemodel_file)
print('ResNetSSD caffe model loaded successfully')

# VIDEO STREAM
name = 'NQToan'
cap = cv.VideoCapture(0)
time.sleep(1.0)
fps = FPS().start()

size = (int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)))
out_fps = 20 
fourcc = cv.VideoWriter_fourcc(*'mp4v') 
writer = cv.VideoWriter()
out_path = base_dir + '05_extracted_webcam/'+ name +'.mp4'
writer.open(out_path, fourcc, out_fps, size, True)

while True:
    _, frame = cap.read()
    origin_h, origin_w = frame.shape[:2]
    blob = cv.dnn.blobFromImage(cv.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    net.setInput(blob)
    detections = net.forward()

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > threshold:
            bounding_box = detections[0, 0, i, 3:7] * np.array([origin_w, origin_h, origin_w, origin_h])
            x_start, y_start, x_end, y_end = bounding_box.astype('int')
            # LABEL FACES
            label = '{0:.2f}%'.format(confidence * 100)
            # BOUNDING BOX
            cv.rectangle(frame, (x_start, y_start), (x_end, y_end),(0, 0, 255), 2)
            # DRAW RECTANGLE
            cv.rectangle(frame, (x_start, y_start-18), (x_end, y_start), (0, 0, 255), -1)
            # PROBABILITY BOX
            cv.putText(frame, label, (x_start+2, y_start-5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    fps.update()
    fps.stop()
    text = "FPS: {:.2f}".format(fps.fps())
    cv.putText(frame, text, (15, int(origin_h * 0.92)), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv.imshow(name, frame)
    writer.write(frame)
    if cv.waitKey(1) & 0xFF == ord("q"): 
        break

writer.release()
cap.release()
cv.destroyAllWindows()