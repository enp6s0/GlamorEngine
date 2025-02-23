#!/usr/bin/python3
#
# GlamorEngine
#
# A standalone virtual background compositor for webcams
# designed to be used standalone in a somewhat air-gapped environment
#
# =====================================================================================================================
import cv2
import time
import multiprocessing
import os
import sys
import numpy as np
import logging
import copy
import traceback
from threading import Thread, Event
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# =====================================================================================================================


class GlamorEngine:

    def __init__(
        self,
        capture_device: str = "/dev/video0",
        gaussian_blur: None | tuple[int, int] = None,
    ) -> None:
        """
        GlamorEngine initializer
        """

        # Get a logger
        self.__logger = logging.getLogger("GlamorEngine")

        # Capture device path
        self.__capture_device_path = str(capture_device).strip()
        self.__logger.info(f"capture device: {self.__capture_device_path}")

        # Root directory, useful for relative asset imports
        self.__my_directory = os.path.dirname(os.path.realpath(__file__))
        self.__logger.info(f"relative root: {self.__capture_device_path}")

        # Gaussian blur?
        self.__gaussian_blur = gaussian_blur
        if self.__gaussian_blur is not None:
            assert type(self.__gaussian_blur) is tuple, "invalid type for Gaussian blur"
            assert len(self.__gaussian_blur) == 2, "invalid tuple for Gaussian blur"

        # Prep background image for when things do go wrong (load and resize to 720p canvas)
        self.__logger.info("loading error background")
        try:
            self.__error_background = cv2.imread(
                f"{self.__my_directory}/assets/technical_difficulties.jpg",
                cv2.IMREAD_COLOR,
            )
            self.__error_background = cv2.resize(
                self.__error_background, (1280, 720), cv2.INTER_AREA
            )
        except Exception as e:
            # If we can't load the error background, generate placeholder
            error_background_bgr = (255, 0, 255)
            self.__error_background = np.zeros((720, 1280, 3), dtype=np.uint8)
            self.__error_background[:] = error_background_bgr
            self.__logger.warning(f"cannot load technical-difficulties background: {e}")

        # Actual virtual background (if available, otherwise use white)
        self.__logger.info("loading virtual background")
        try:
            self.__virtual_background = cv2.imread(
                f"{self.__my_directory}/assets/virtual_background.jpg", cv2.IMREAD_COLOR
            )
            self.__virtual_background = cv2.resize(
                self.__virtual_background, (1280, 720), cv2.INTER_AREA
            )
        except Exception as e:
            # If we can't load the error background, generate placeholder
            virtual_background_bgr = (255, 255, 255)
            self.__virtual_background = np.zeros((720, 1280, 3), dtype=np.uint8)
            self.__virtual_background[:] = virtual_background_bgr
            self.__logger.warning(f"cannot load virtual background: {e}")

        # Copy error background into framebuffer, and set error text to "initializing"
        self.__framebuffer_text = "GlamorEngine is initializing..."
        self.__framebuffer = copy.copy(self.__error_background)

        # MediaPipe selfie segmentation, thing doing actual detection & segmentation
        self.__logger.info("initializing segmentation")
        self.__model_path = os.path.join(
            self.__my_directory, "models", "selfie_segmenter_landscape.tflite"
        )

        # MediaPipe handles
        BaseOptions = mp.tasks.BaseOptions
        self.ImageSegmenter = (
            mp.tasks.vision.ImageSegmenter
        )  # registered globally since we need this later
        ImageSegmenterOptions = mp.tasks.vision.ImageSegmenterOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        # MediaPipe options
        self.__mp_options = ImageSegmenterOptions(
            base_options=BaseOptions(model_asset_path=self.__model_path),
            running_mode=VisionRunningMode.IMAGE,
            output_category_mask=True,
        )

        # Exit event, so we can properly kill threads
        self.__exit_event = Event()

        # Internal framerate delay
        # (there's no need to run this for more than 30fps, so we delay and
        # release the GIL such that other threads can do their thing...)
        self.__fps_delay = 0.0334

        # Start framebuffer thread
        self.__logger.info("starting framebuffer thread")
        self.__fb_write_thread = Thread(target=self.__framebuffer_write)
        self.__fb_write_thread.start()

        # Start capture thread
        self.__logger.info("starting camera thread")
        self.__video_capture_thread = Thread(target=self.__capture_thread)
        self.__video_capture_thread.start()

    # =================================================================================================================

    def close(self) -> None:
        """
        Gracefully terminate GlamorEngine
        """
        self.__exit_event.set()
        self.__fb_write_thread.join()
        self.__video_capture_thread.join()

    # =================================================================================================================

    def __capture_thread(self) -> None:
        """
        Main capture/processing thread
        """
        # Run this forever
        while not self.__exit_event.is_set():

            try:
                # Open capture device
                cam = cv2.VideoCapture(self.__capture_device_path)

                # With MediaPipe segmenter...
                with self.ImageSegmenter.create_from_options(
                    self.__mp_options
                ) as segmenter:

                    # Try to continously capture
                    while not self.__exit_event.is_set():
                        ret, frame = cam.read()

                        # Check that we can get frame
                        if frame is None:

                            # Can't get frame, does the device exist?
                            if not os.path.exists(self.__capture_device_path):
                                raise Exception(
                                    f"E002: capture device gone AWOL: {self.__capture_device_path} [{int(time.time())}]"
                                )
                            else:
                                raise Exception(
                                    f"E003: framebuffer is starving [{int(time.time())}]"
                                )

                        # Resize incoming frame to 1280x720 to match our internal canvas
                        frame = cv2.resize(frame, (1280, 720), cv2.INTER_AREA)

                        # Segment frame and get the mask
                        mp_frame = mp.Image(
                            image_format=mp.ImageFormat.SRGB, data=frame
                        )
                        mask = segmenter.segment(mp_frame).category_mask.numpy_view()

                        # Smooth out mask
                        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
                        (thresh, binRed) = cv2.threshold(
                            mask, 128, 255, cv2.THRESH_BINARY
                        )
                        mask = cv2.morphologyEx(
                            mask, cv2.MORPH_OPEN, kernel, iterations=3
                        )

                        # Do the masking
                        replacement = [255, 255, 255]
                        frame[mask == 255] = replacement

                        # TODO: actually compose the virtual background in here

                        # If Gaussian blur is enabled, do so on the image
                        if self.__gaussian_blur is not None:
                            frame = cv2.GaussianBlur(
                                frame, self.__gaussian_blur, cv2.BORDER_DEFAULT
                            )

                        # Load final output into frame buffer
                        self.__framebuffer_text = ""
                        self.__framebuffer = copy.copy(frame)

                    # Release GIL to let other threads run too
                    time.sleep(0)

            except Exception as e:
                # Hmm, something went wrong here
                self.__framebuffer = copy.copy(self.__error_background)
                self.__framebuffer_text = f"{e}"
                self.__logger.error(e)
                self.__logger.error(traceback.format_exc())

                # Try to release the camera, but if that fails, not a big deal
                try:
                    cam.release()
                except Exception:
                    pass

                # Wait a bit before trying again, hopefully whatever's at fault
                # would have fixed itself before then
                time.sleep(0.5)

        # If we're here, time to exit
        self.__logger.debug("capture thread terminating")

        # Try to release the camera, but if that fails, again, not a big deal
        try:
            cam.release()
        except Exception:
            pass

        return

    # =================================================================================================================

    def __framebuffer_write(self) -> None:
        """
        Thread for writing out to raw framebuffer
        """

        # Run this forever
        while not self.__exit_event.is_set():

            # Copy raw frame buffer
            out_frame = self.__framebuffer.copy()

            # If framebuffer is none, skip writing this, as there's nothing we can do
            if out_frame is None:
                continue

            # Add text, if one is specified
            if self.__framebuffer_text is not None and self.__framebuffer_text != "":
                out_frame = cv2.putText(
                    out_frame,
                    f"{self.__framebuffer_text}",
                    (30, 700),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (40, 40, 40),
                    2,
                    cv2.LINE_AA,
                )

            # Figure out framebuffer size
            with open("/sys/class/graphics/fb0/modes", "r") as f:
                fb_mode = f.read().strip()  # gives something like: U:1920x1080p-0

            # Parse framebuffer size
            try:
                fb_width = int(str(fb_mode.split("x", 1)[0].split(":", 1)[-1]).strip())

                # Height is a bit trickier, we need to find first character
                # that isn't a number
                height_str = str(fb_mode.split("x", 1)[1])
                fb_height = 0
                for i, c in enumerate(height_str):
                    if not c.isdigit():
                        fb_height = int(height_str[: i - 1])

            # If this doesn't work, we use defaults
            except Exception as e:
                self.__logger.error(f"framebuffer size query exception: {e}")
                fb_width = 1280
                fb_height = 720

            # Convert output frame color space
            out_frame = cv2.cvtColor(out_frame, cv2.COLOR_BGR2BGRA)

            # Resize output frame to framebuffer size
            out_frame = cv2.resize(out_frame, (fb_width, fb_height), cv2.INTER_AREA)

            # Write to framebuffer
            try:
                with open("/dev/fb0", "wb+") as buf:
                    buf.write(out_frame)
            except Exception as e:
                self.__logger.error(f"framebuffer write error: {e}")

            # Release GIL for other threads
            time.sleep(self.__fps_delay)

        # Thread should be terminated
        self.__framebuffer = None
        self.__framebuffer_text = None
        self.__logger.debug("framebuffer thread terminating")
        return

    # =================================================================================================================

    def hologram(self, frame: np.array) -> np.array:
        # Hologram effect from https://elder.dev/posts/open-source-virtual-background/

        holo = cv2.applyColorMap(frame, cv2.COLORMAP_WINTER)
        band_length, band_gap = 2, 3

        for y in range(holo.shape[0]):

            if y % (band_length + band_gap) < band_length:

                holo[y, :, :] = holo[y, :, :] * np.random.uniform(0.1, 0.3)

        def shift_img(img, dx, dy):
            img = np.roll(img, dy, axis=0)
            img = np.roll(img, dx, axis=1)

            if dy > 0:
                img[:dy, :] = 0

            elif dy < 0:
                img[dy:, :] = 0

            if dx > 0:
                img[:, :dx] = 0

            elif dx < 0:
                img[:, dx:] = 0

            return img

        holo2 = cv2.addWeighted(holo, 0.2, shift_img(holo.copy(), 5, 5), 0.8, 0)
        holo2 = cv2.addWeighted(holo2, 0.4, shift_img(holo.copy(), -5, -5), 0.6, 0)
        holo_done = cv2.addWeighted(frame, 0.5, holo2, 0.6, 0)

        return holo_done


# =====================================================================================================================

if __name__ == "__main__":

    # Set up logging environment
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)

    # Capture device
    try:
        capture_device = str(sys.argv[1]).strip()
    except IndexError:
        capture_device = "/dev/video0"

    # Start engine
    try:
        engine = GlamorEngine(capture_device=capture_device, gaussian_blur=(1, 1))
    except KeyboardInterrupt:
        engine.close()
