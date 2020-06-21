# import the necessary packages
from imutils.video import FPS
import numpy as np
import argparse
import matplotlib.pyplot as plt
import cv2
import os

use_gpu = 1
webcam = 1
expected_confidence = 0.3
threshold = 0.1
show_output = 1
save_output = 1
kernel = np.ones((5,5),np.uint8)


# derive the paths to the Mask R-CNN weights and model configuration
weightsPath = "mask-rcnn-coco/frozen_inference_graph.pb"
configPath = "mask-rcnn-coco/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt"

# load our Mask R-CNN trained on the COCO dataset (90 classes) from disk
print("[INFO] loading Mask R-CNN from disk...")
net = cv2.dnn.readNetFromTensorflow(weightsPath, configPath)

# check if we are going to use GPU
if use_gpu:
	# set CUDA as the preferable backend and target
	print("[INFO] setting preferable backend and target to CUDA...")
	net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
	net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# initialize the video stream and pointer to output video file, then
# start the FPS timer
print("[INFO] accessing video stream...")
if webcam:
	cap = cv2.VideoCapture(0)
else:
	cap = cv2.VideoCapture('humans.mp4')

writer = None
fps = FPS().start()

print("[INFO] background recording...")
for _ in range(30):
	_,bg = cap.read()
print("[INFO] background recording done...")

fourcc = cv2.VideoWriter_fourcc(*"MJPG")
writer = cv2.VideoWriter('output.avi', fourcc, 20,(bg.shape[1], bg.shape[0]), True)


# loop over frames from the video file stream
while True:
	grabbed, frame = cap.read()
	cv2.imshow('org',frame)
	# if the frame was not grabbed, then we have reached the end of the stream
	if not grabbed:
		break

	# construct a blob from the input frame and then perform a
	# forward pass of the Mask R-CNN, giving us (1) the bounding box
	# coordinates of the objects in the image along with (2) the
	# pixel-wise segmentation for each specific object
	blob = cv2.dnn.blobFromImage(frame, swapRB=True, crop=False)
	net.setInput(blob)
	(boxes, masks) = net.forward(["detection_out_final","detection_masks"])
	# loop over the number of detected objects
	for i in range(0, boxes.shape[2]):
		# extract the class ID of the detection along with the confidence (i.e., probability) associated with the prediction
		classID = int(boxes[0, 0, i, 1])
		if classID!=0:continue
		confidence = boxes[0, 0, i, 2]

		if confidence > expected_confidence:
			# scale the bounding box coordinates back relative to the size of the frame and then compute the width and the height of the bounding box
			(H, W) = frame.shape[:2]
			box = boxes[0, 0, i, 3:7] * np.array([W, H, W, H])
			(startX, startY, endX, endY) = box.astype("int")
			boxW = endX - startX
			boxH = endY - startY

			# extract the pixel-wise segmentation for the object, resize the mask such that it's the same dimensions of
			# the bounding box, and then finally threshold to create a *binary* mask
			# shape of masks is [100,90,15,15]
			# means each of the 90 classes is having a mask of 15X15 and there are 100 detections as usual 
			mask = masks[i, classID]
			mask = cv2.resize(mask, (boxW, boxH),interpolation=cv2.INTER_CUBIC)
			mask = (mask > threshold)
			bwmask = np.array(mask,dtype=np.uint8) * 255
			bwmask = np.reshape(bwmask,mask.shape)
			bwmask = cv2.dilate(bwmask,kernel,iterations=1)

			# take out that portion out from the bg image and paste it into the current frame
			frame[startY:endY, startX:endX][np.where(bwmask==255)] = bg[startY:endY, startX:endX][np.where(bwmask==255)]

	# check to see if the output frame should be displayed to our screen
	if show_output:
		cv2.imshow("Frame", frame)

		if cv2.waitKey(1) ==27:
			break

	# if the video writer is not None, write the frame to the output video file
	if save_output:
		writer.write(frame)

	# update the FPS counter
	fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))