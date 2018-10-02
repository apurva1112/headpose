from imutils.video import VideoStream
import numpy as np
import imutils
import time
from sys import exit
import cv2
from keras.applications.mobilenetv2 import MobileNetV2
from keras.layers import Dense, Flatten
from keras.models import Model
from math import cos, sin


# Some parameters
pad = 20  # Amount of padding to keep on all sides of the face while cropping
face, frame = None, None

shape = (160, 160, 3)
base_model = MobileNetV2(input_shape=shape, include_top=False)
x = base_model.output
x = Flatten()(x)
x = Dense(100, activation='relu')(x)
x = Dense(50, activation='relu')(x)
x = Dense(100, activation='relu')(x)
predictions = Dense(3, activation=None)(x)
model = Model(inputs=base_model.input, outputs=predictions)
model.load_weights('./weights.h5')


def draw_axis(img, yaw, pitch, roll, size=50):

    pitch = pitch * np.pi / 180
    yaw = -(yaw * np.pi / 180)
    roll = roll * np.pi / 180

    height, width = img.shape[:2]
    tdx = width / 2
    tdy = height / 2

    x1 = size * (cos(yaw) * cos(roll)) + tdx
    y1 = size * (cos(pitch) * sin(roll) +
                 cos(roll) * sin(pitch) * sin(yaw)) + tdy

    x2 = size * (-cos(yaw) * sin(roll)) + tdx
    y2 = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll))\
        + tdy

    x3 = size*250 * (sin(yaw)) + tdx
    y3 = size*250 * (-cos(yaw) * sin(pitch)) + tdy

    #cv2.line(img, (int(tdx), int(tdy)), (int(x1), int(y1)), (0, 0, 255), 3)
    #cv2.line(img, (int(tdx), int(tdy)), (int(x2), int(y2)), (0, 255, 0), 3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x3), int(y3)), (255, 0, 0), 2)

    return img


net = cv2.dnn.readNetFromCaffe("./deploy.prototxt.txt",
                               "./res10_300x300_ssd_iter_140000.caffemodel")

# Initialize the video stream
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()

# loop over the frames from the video stream
while True:
    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 400 pixels
    frame = vs.read()
    frame = imutils.resize(frame, width=400)

    # grab the frame dimensions and convert it to a blob
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)), 1.0,
        (300, 300), (104.0, 177.0, 123.0)
        )

    # pass the blob through the network and obtain the detections and
    # predictions
    net.setInput(blob)
    detections = net.forward()

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the
        # prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if confidence < 0.35:
            continue

        # compute the (x, y)-coordinates of the bounding box for the
        # object
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        face = frame[(startY-pad):(endY+pad), (startX-pad):(endX+pad)]
        face = cv2.resize(face, (160, 160))  # Input size
        face_ = face[None, ...]/255.
        pre = model.predict(face_)
        pitch, yaw, roll = pre[0]
        print(pre)
        face = draw_axis(face, yaw, pitch, roll)
        # draw the bounding box of the face along with the associated
        # probability
        text = "{:.2f}%".format(confidence * 100)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
        cv2.putText(
            frame, text, (startX, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45, (0, 0, 255), 2)

    if face is not None:
        cv2.imshow("Axis", face)
    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
    elif key == 27:
        break
    elif not cv2.getWindowProperty("Frame", cv2.WND_PROP_VISIBLE):
        break


# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
