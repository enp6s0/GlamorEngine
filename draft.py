#!/usr/bin/python3
#
# CameraEngine
#
# A standalone compositor for webcams
# designed to be used standalone in a somewhat air-gapped environment
#
import cv2, time, multiprocessing, os
from multiprocessing import Process, Value, Array, Lock

print("CameraEngine v0.1 is starting...")

# Get script directory
scriptDir = os.path.dirname(os.path.realpath(__file__))

# Prep background image for when things do go wrong
# This is expected to be 1280 x 720!
errorBG = cv2.imread(f"{scriptDir}/assets/technical_difficulties.jpg", cv2.IMREAD_COLOR)

# Frame and error code
frame = Array("d", lock = True)
errorCode = Value("i", lock = True)


def camThread(deviceHandle):
    global frameGood
    global errorMessage
    global lastFrameAt
    global cameraFrame

    # Recovery loop
    while True:
        # Try to capture frame
        try:

            # Open capture device
            cam = cv2.VideoCapture(0)

            # Try reading
            while True:

                ret, frame = cam.read()

                # Check that we can get frame
                if ret is not True:
                    raise Exception("E001: framebuffer is starving")

                # Resize frame to standard 720p
                frame = cv2.resize(frame, (1280, 720), cv2.INTER_AREA)

                # Frame is good, load frame
                try:
                    cameraFrameLock.acquire()
                    cameraFrame = frame
                    lastFrameAt = time.time()
                    frameGood = True
                    errorMessage = ""
                finally:
                    cameraFrameLock.release()

        except Exception as e:
            frameGood = False
            errorMessage = f"{e}"


def outputThread():
    global errorMessage

    while True:
        # If frame is not good, technical difficulties
        if(frameGood != True):
            print("BLAH")
            tdOutFrame = errorBG.copy()
            tdOutFrame = cv2.putText(tdOutFrame, f"{errorMessage}", (30, 700), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (80, 80, 255), 2, cv2.LINE_AA)
            cv2.imshow("output", tdOutFrame)
            cv2.waitKey()
        else:
            # Is the frame fresh?
            if(time.time() > lastFrameAt + 1):
                # Nope...
                frameGood = False
                errorMessage = "E002: rotten frames"

                # Will pick up the error message in the next loop
            else:
                # Yes, display it
                try:
                    cameraFrameLock.acquire()
                    cv2.imshow("output", cameraFrame)
                    cv2.waitKey()
                finally:
                    cameraFrameLock.release()

# Start capture process and output/watchdog/display thread
cpThread = Process(target = camThread, args = ("/dev/video0", ))
dpThread = Process(target = outputThread)

cpThread.start()
dpThread.start()

exit(0)

# Main loop, so whenever there is an exception, we return here and try again
while True:

    # OpenCV fullscreen named window that we can output stuff to later
    cv2.namedWindow("output", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("output",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)

    # cv2.imshow("output", img)

    try:

        # Open capture device
        cam = cv2.VideoCapture(0)

        # Try reading
        while True:

            ret, frame = cam.read()

            # Check that we can get frame
            if ret is not True:
                raise Exception("E001: framebuffer is starving")
            
            # Resize whatever to 720p, our standard
            frame = cv2.resize(frame, (1280, 720), cv2.INTER_AREA)

            # Display
            cv2.imshow("output", frame)
            cv2.waitKey()

    except Exception as e:

        # Try to release cameras, we'll capture it again later
        try:
            cam.release()
        except:
            # Eh, doesn't work but who cares
            pass

        # Technical difficulties
        tdOutFrame = errorBG.copy()
        tdOutFrame = cv2.putText(tdOutFrame, f"{e}", (30, 700), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (80, 80, 255), 2, cv2.LINE_AA)
        cv2.imshow("output", tdOutFrame)
        cv2.waitKey()
        # print(f"Exception: {e}")