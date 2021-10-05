# import the necessary packages
from imutils.video import VideoStream
from imutils.video import WebcamVideoStream
from imutils.video import FPS
import argparse
import imutils
import time
import cv2
import numpy as np
import logging

CAFFE_MODEL = 'face/model/res10_300x300_ssd_iter_140000.caffemodel'
PROTOTEXT = 'face/model/deploy.prototxt.txt'
SHOW_FRAME = True
DO_PREDICT = False
SHOW_FPS = True
CONFIDENT_THRESHOLD = 0.5
MAX_NUM_FRAME = 500
WIDTH = 1000

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s :: %(levelname)s :: %(message)s')

# construct the argument parse and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-c", "--confidence", type=float, default=0.5,
#                 help="minimum probability to filter weak detections")
# args = vars(ap.parse_args())

# Load model
# logging.info("Loading Model")
net = cv2.dnn.readNetFromCaffe(PROTOTEXT, CAFFE_MODEL)

# initialize the video stream and allow the camera sensor to warm up
# logging.info("Starting Video Stream")
vs = VideoStream(src=0).start()
fps = FPS().start()
time.sleep(2.0)

# For real-time FPS calculation
prev_frame_time = 0
new_frame_time = 0

logging.info("Prediction: {}, Show Frame: {}, Width: {}".format(
    DO_PREDICT, SHOW_FRAME, WIDTH))

# loop over the frames from the video stream
while True:
    if SHOW_FPS:
        new_frame_time = time.time()
    # Feed the video stream
    frame = vs.read()

    # Frame Tranformation
    frame = imutils.resize(frame, width=WIDTH)
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # frame = np.dstack([frame, frame, frame])

    # convert frame to blob
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))

    # Face Detection
    if DO_PREDICT:
        net.setInput(blob)
        detections = net.forward()
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence < CONFIDENT_THRESHOLD:
                continue
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            text = "{:.2f}%".format(confidence * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                          (0, 0, 255), 2)
            cv2.putText(frame, text, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

    # Calculating the fps
    if SHOW_FPS:
        frame_ps = 1/(new_frame_time-prev_frame_time)
        prev_frame_time = new_frame_time
        cv2.putText(frame, str(int(frame_ps)), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Show frame
    if SHOW_FRAME:
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    if MAX_NUM_FRAME > 0:
        if fps._numFrames >= MAX_NUM_FRAME:
            break

    fps.update()

# Clean up
fps.stop()
logging.info("elasped time: {:.2f}".format(fps.elapsed()))
logging.info("approx. FPS: {:.2f}".format(fps.fps()))
vs.stop()
cv2.destroyAllWindows()
