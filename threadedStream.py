from __future__ import print_function
# from imutils.video import WebcamVideoStream
# from imutils.video import FPS
from video import video
from fps import FPS
import argparse
import imutils
import cv2

def threadedTest():
	# created a *threaded* video stream, allow the camera sensor to warmup,
	# and start the FPS counter
	print("[INFO] sampling THREADED frames from webcam...")
	vs = video(src="rtsp://admin:admin@192.168.1.137:30032").start() #using diff thread to do I/O for images
	# fps = FPS().start()
    frame = vs.read()
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # fps.update()
	
	# fps.stop()
	# print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
	# print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
	return frame
	# do a bit of cleanup
	cv2.destroyAllWindows()
	vs.stop()
