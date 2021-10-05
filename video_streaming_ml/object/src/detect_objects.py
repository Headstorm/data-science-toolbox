import cv2
import time
import imutils
import argparse
import numpy as np
import logging
from imutils.video import FPS
from imutils.video import VideoStream

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s :: %(levelname)s :: %(message)s')

#Constructing Argument Parse to input from Command Line
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", help = 'Path to prototxt', default = 'object/model/SSD_MobileNet_prototxt.txt')
ap.add_argument("-m", "--model", help = 'Path to model weights', default = 'object/model/SSD_MobileNet.caffemodel')
ap.add_argument("-c", "--confidence", type = float, default = 0.7)
args = vars(ap.parse_args())

#Initialize Objects and corresponding colors which the model can detect
labels = ["background", "aeroplane", "bicycle", "bird", 
"boat","bottle", "bus", "car", "cat", "chair", "cow", 
"diningtable","dog", "horse", "motorbike", "person", "pottedplant", 
"sheep","sofa", "train", "tvmonitor"]
colors = np.random.uniform(0, 255, size=(len(labels), 3))

#Loading Caffe Model
print('[Status] Loading Model...')
nn = cv2.dnn.readNetFromCaffe(args['prototxt'], args['model'])

#Initialize Video Stream
print('[Status] Starting Video Stream...')
vs = VideoStream(src=0, framerate=120, resolution=(1280,720)).start()
time.sleep(2.0)
fps = FPS().start()

#Loop Video Stream
while True:

    #Resize Frame to 400 pixels
    frame = vs.read()
    frame = imutils.resize(frame, width=1280)
    (h, w) = frame.shape[:2]

    #Converting Frame to Blob
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 
    	0.007843, (300, 300), 127.5)

    #Passing Blob through network to detect and predict
    nn.setInput(blob)
    detections = nn.forward()


    #Loop over the detections
    for i in np.arange(0, detections.shape[2]):

	#Extracting the confidence of predictions
        confidence = detections[0, 0, i, 2]

        #Filtering out weak predictions
        if confidence > args["confidence"]:
            
            #Extracting the index of the labels from the detection
            #Computing the (x,y) - coordinates of the bounding box        
            idx = int(detections[0, 0, i, 1])

            #Extracting bounding box coordinates
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            #Drawing the prediction and bounding box
            label = "{}: {:.2f}%".format(labels[idx], confidence * 100)
            cv2.rectangle(frame, (startX, startY), (endX, endY), colors[idx], 2)

            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, label, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[idx], 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    
    fps.update()

fps.stop()

print("[Info] Elapsed time: {:.2f}".format(fps.elapsed()))
print("[Info] Approximate FPS:  {:.2f}".format(fps.fps()))

cv2.destroyAllWindows()
vs.stop()
