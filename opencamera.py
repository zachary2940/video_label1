import numpy as np
import cv2
import time

start_time = time.time()
cap = cv2.VideoCapture("http://192.168.1.152:8080")
print("--- %s seconds ---" % (time.time() - start_time))

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # Our operations on the frame come here
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # frame = frame[300:599, 700:999]
    # frame= cv2.resize(frame,(299,299),3)
    # print type(frame)
    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()