#!/usr/bin/python3
#
# GlamorEngine
#
# A standalone virtual background compositor for webcams
# designed to be used standalone in a somewhat air-gapped environment
#
import cv2, time, multiprocessing, os
print("GlamorEngine v0.1 is starting...")

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

# Load virtual background
virtualBG = cv2.imread(f"{scriptDir}/assets/backgrounds/bg1.jpg", cv2.IMREAD_COLOR)
virtualBG = cv2.resize(virtualBG, (1280, 720), cv2.INTER_AREA)

# Import MediaPipe
import mediapipe as mp
import numpy 

def hologram(frame):
    # Hologram effect from https://elder.dev/posts/open-source-virtual-background/

    holo = cv2.applyColorMap(frame, cv2.COLORMAP_WINTER)
    bandLength, bandGap = 2, 3

    for y in range(holo.shape[0]):

        if y % (bandLength+bandGap) < bandLength:

            holo[y,:,:] = holo[y,:,:] * np.random.uniform(0.1, 0.3)


    def shift_img(img, dx, dy):
        img = np.roll(img, dy, axis=0)
        img = np.roll(img, dx, axis=1)

        if dy>0:
            img[:dy, :] = 0

        elif dy<0:
            img[dy:, :] = 0

        if dx>0:
            img[:, :dx] = 0

        elif dx<0:
            img[:, dx:] = 0

        return img


    holo2 = cv2.addWeighted(holo, 0.2, shift_img(holo.copy(), 5, 5), 0.8, 0)
    holo2 = cv2.addWeighted(holo2, 0.4, shift_img(holo.copy(), -5, -5), 0.6, 0)

    holo_done = cv2.addWeighted(frame, 0.5, holo2, 0.6, 0)

    return holo_done

# Main loop, so whenever there is an exception, we return here and try again
while True:

    try:
        # Selfie segmentation
        mpss = mp.solutions.selfie_segmentation
        ss = mpss.SelfieSegmentation(model_selection = 0)

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
            results = ss.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
            black = np.zeros(frame.shape, dtype = np.uint8)
            white = np.zeros(frame.shape, dtype = np.uint8)
            white[:] = (255, 255, 255)

            # Create mask alone
            mask = np.where(condition, white, black)

            # Smooth out mask
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
            (thresh, binRed) = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=3)

            # Now do a hologram effect and masking all in one go :)
            cameraOutput = np.where(mask, hologram(frame), black)

            # Lay it on top of the virtual background
            invertedMask = 1 - mask
            croppedBG = np.where(invertedMask, virtualBG, black)

            output = cv2.addWeighted(croppedBG, 1.0, cameraOutput, 0.98, 0)

            # Display
            cv2.imshow("output", output) # frame
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