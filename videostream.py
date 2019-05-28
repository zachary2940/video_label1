import cv2
import socket
import numpy as np
#rtsp://admin:admin@192.168.1.137:30022
class FileVideoStream:

    def __init__(self, path):
        """Initialize the file video stream along with the boolean used to indicate if the thread should be stopped or not"""
        # self.stream = cv2.VideoCapture(path)
        self.stream=cv2.VideoCapture(path)
        # open_cam_rtsp(path, 1280, 720, 10)
        # cv2.VideoCapture(path)
        self.stopped = False
        self.frame = 1
        self.width = int(round((self.stream.get(3))))
        self.height = int(round((self.stream.get(4))))

    def start(self):
        """Start a thread to read frames from the file video stream"""
        t = Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        while True:  # keep looping infinitely
            # read the next frame from the file
            (grabbed, frame) = self.stream.read()
            if(not grabbed):
                self.stop()
                print "Camera stopped!"
                '''
                # Create a socket here to see if the program has to be stopped.
                sock_quit = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                try:
                    sock_quit.connect((HOST, PORT_QUIT))
                    data = b"rtsp"
                    sock_quit.send(data)
                except socket.error as e:
                    print("Socket error: %s" % str(e))
                finally:
                    sock_quit.close()
                    time.sleep(2)
                '''
                return
            self.frame = frame.copy()
            # if the `grabbed` boolean is `False`, then we have
            # reached the end of the video file

    def read(self):
        """Return next frame in the queue"""
        return self.frame

    def stop(self):
        """Indicate that the thread should be stopped"""
        self.stopped = True
        return

        