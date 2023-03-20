#!/usr/bin/python3
#
# CameraEngine
#
# A standalone compositor for webcams
# designed to be used standalone in a somewhat air-gapped environment
#
import cv2, time, multiprocessing, os
print("CameraEngine v0.1 is starting...")

# specify device
DEVICE = "/dev/v4l/by-id/usb-046d_081b_8D13BC60-video-index0"
print(f"Using device: {DEVICE}")

# Get script directory
scriptDir = os.path.dirname(os.path.realpath(__file__))

# Prep background image for when things do go wrong
# This is expected to be 1280 x 720!
errorBG = cv2.imread(f"{scriptDir}/assets/technical_difficulties.jpg", cv2.IMREAD_COLOR)

# OpenCV fullscreen named window that we can output stuff to later
cv2.namedWindow("output", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("output",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
time.sleep(0.1)

# Display loading screen
tdOutFrame = errorBG.copy()
tdOutFrame = cv2.putText(tdOutFrame, f"Initializing VideoEngine...", (30, 700), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (40, 40, 40), 2, cv2.LINE_AA)
cv2.imshow("output", tdOutFrame)
cv2.waitKey(5)

# Import TensorFlow and Bodypix prerequisites (will take some time)
from pathlib import Path
import numpy as np
import tensorflow as tf
from tf_bodypix.api import download_model, load_model, BodyPixModelPaths

bpxModel = None
def loadBodypixModel():
    # Load BodyPix model
    global bpxModel
    try:
        # tfjs version
        #bpxModel = load_model(f"{scriptDir}/models/8ba301b16e59fd7bda330880a9d70e58--tfjs-models-savedmodel-bodypix-mobilenet-float-050-model-stride16")

        # tflite version
        bpxModel = load_model(f"{scriptDir}/models/tflite/mobilenet-float-multipler-050-stride16-float16.tflite")

        return True
    except Exception as e:
        print(f"Bodypix load error: {e}")
        bpxModel = None
        return f"{e}"

# Main loop, so whenever there is an exception, we return here and try again
while True:

    # OpenCV fullscreen named window that we can output stuff to later
    # cv2.namedWindow("output", cv2.WND_PROP_FULLSCREEN)
    # cv2.setWindowProperty("output",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
    # cv2.imshow("output", img)

    try:

        # Load bodypix masking (glamor) model
        bpxLoad = loadBodypixModel()

        # Did we have the bodypix model?
        if bpxModel is None:
            raise Exception(f"E007: glamor: {bpxLoad}")

        # Open capture device
        cam = cv2.VideoCapture(DEVICE)

        # Try reading
        while True:

            ret, frame = cam.read()

            # Check that we can get frame
            if ret is not True:

                # Can't get frame, does the device exist?
                if(not os.path.exists(DEVICE)):
                    raise Exception(f"E002: capture device gone AWOL: {DEVICE}")
                else:
                    raise Exception("E003: framebuffer is starving")

            # Resize whatever to 720p, our standard
            frame = cv2.resize(frame, (1280, 720), cv2.INTER_AREA)

            # Pass along frame to bodypix for mask processing
            bpxResult = bpxModel.predict_single(frame)
            bpxMask = bpxResult.get_mask(threshold = 0.75).numpy().astype(np.uint8)

            # Bodypix masking
            masked = cv2.bitwise_and(frame, frame, mask = bpxMask)

            # Display
            cv2.imshow("output", masked) # frame
            if cv2.waitKey(1) & 0xFF == ord("q"):
                exit(0)

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

        if cv2.waitKey(1) & 0xFF == ord("q"):
            exit(0)

        # Wait a bit, hopefully this gives time for the underlying fault
        # condition to fix itself
        time.sleep(0.5)

        # print(f"Exception: {e}")
