import numpy as np
import cv2 as cv
import os

threshold = 0.5  # CONFIDENCE THRESHOLD
base_dir = os.path.dirname(__file__)

# LOAD MODEL
prototxt_file = '01_model/Resnet_SSD_deploy.prototxt'
caffemodel_file = '01_model/Res10_300x300_SSD_iter_140000.caffemodel'
net = cv.dnn.readNetFromCaffe(prototxt_file, caffeModel=caffemodel_file)
print('MobileNetSSD caffe model loaded successfully')
count = 0

# LOAD IMG
img_path = '02_images/test3.jpg'
filename_img = os.path.basename(img_path).split('.')[0]
image = cv.imread(img_path)
origin_h, origin_w = image.shape[:2]
blob = cv.dnn.blobFromImage(cv.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
net.setInput(blob)

# DETECTION
detections = net.forward()
print('Face detection accomplished')

# FIND FACES
for i in range(detections.shape[2]):
    confidence = detections[0, 0, i, 2]
    if confidence > threshold:
        # BOUNDING BOX
        count += 1
        bounding_box = detections[0, 0, i, 3:7] * np.array([origin_w, origin_h, origin_w, origin_h])
        x_start, y_start, x_end, y_end = bounding_box.astype('int')
 
        # DRAW RECTANGLE AND SAVE FILES
        frame = image[y_start:y_end, x_start:x_end]
        cv.imwrite(base_dir + '03_extracted_faces/' + filename_img + '_' + str(i) +'.jpg', frame)

        label = '{0:.2f}%'.format(confidence * 100)
        cv.rectangle(image, (x_start, y_start), (x_end, y_end), (255, 0, 0), 2)
        cv.rectangle(image, (x_start, y_start), (x_end, y_end), (255, 0, 0), 2)
        cv.rectangle(image, (x_start, y_start - 18), (x_end, y_start), (255, 0, 0), -1) 
        cv.putText(image, label, (x_start+2, y_start-5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
cv.imwrite(base_dir + '04_extracted_images/' +filename_img + '_Extracted' + '.jpg', image)
print(filename_img + " converted successfully")
cv.imshow(filename_img, image)

print("Saved " + str(count) + ' extracted faces' + ' to 03_extracted_faces')
print("Saved " + filename_img + '_Extracted' + '.jpg' + ' to 04_extracted_images')

cv.waitKey(0)
cv.destroyAllWindows()
