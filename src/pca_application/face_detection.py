import cv2
import numpy as np

# face detection deep neural network
net = cv2.dnn.readNetFromCaffe(prototxt='./data/model/deploy.prototxt',
                               caffeModel='./data/model/res10_300x300_ssd_iter_140000.caffemodel')

# read my face images
image_paths = ['./data/att_faces/input1.jpg', './data/att_faces/input2.jpg', './data/att_faces/input3.jpg']
images = []

for path in image_paths:
    image = cv2.imread(path)
    images.append(image)

# detect, crop and save my face
for i, image in enumerate(images):
    h, w = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image=cv2.resize(image, (300, 300)), scalefactor=1.0, size=(300, 300),
                                 mean=(103.93, 116.77, 123.68))
    net.setInput(blob)
    detection = net.forward()[0, 0, 0, :]
    box = detection[3:7] * np.array([w, h, w, h])
    x_start, y_start, x_end, y_end = box.astype('int')
    face = image[y_start:y_end, x_start:x_end]
    face = cv2.resize(src=face, dsize=(46, 56))
    face = cv2.cvtColor(src=face, code=cv2.COLOR_BGR2GRAY)
    # cv2.imshow('face', face)
    # cv2.waitKey(0)
    cv2.imwrite('./data/att_faces/my_face_{}.jpg'.format(i + 1), face)
