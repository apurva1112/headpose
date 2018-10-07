from imutils.video import VideoStream, FPS
from imutils import resize
import numpy as np
import cv2
import os

# TODO:
# 1.Implement the function to automatically reset the
#   bounding box for the detected object after a certain interval
#   (in no. of frames, or preferrably, no. of seconds)


class Tracker():
    """
    Tracker(
        self, detector=None, confidence=0.4, tracker='KCF',
        refresh_interval=40, video=None, width=400, window=None
            )

    Creates a `Tracker` object, designed to automatically detect and
    subsequently track the motion of any single object.

    Parameters
    ----------
    -`detector` : list/tuple, optional
        A 2-tuple in the format (prototxt, caffemodel).
        `prototxt` is the path to the .prototxt file with description
        of network architecture.
        `caffemodel` is the path to the learned Caffe model to be used
        for detecting the object.
        If not provided, lets the user select a ROI by himself.

    -`confidence` : float (0 < confidence < 1), optional
        The threshold confidence level to select a bounding box. Is
        ignored when a detector is not provided.
        A real number between 0 and 1. Defaults to 0.4. Any bounding box
        having confidence less than this value will be rejected.

    -`refresh_interval` : int, optional
        The no. of frames after which to detect the object again and reset the
        bounding box. Ignored if a `detector` is not provided. Defaults to 40.
        Helps prevent the case wherein the tracker loses the object.
        
        *Yet to be implemented

    -`tracker` : {'KCF','CSRT','MIL','Boosting','MedianFlow','TLD','MOSSE'},
                optional
        The name of the tracker to use. Defaults to KCF.

    `video` : str, optional
        If a valid video path is provided, uses the video for detecting
        and tracking the objecting. Otherwise, uses webcam.

    `width` : int, optional
        The width in pixels to which to resize the frame. Used for reducing
        the computation. Lower the width, lower the computation time,
        but also harder for the detector to detect objects in the image.
        Defaults to 400 px.

    `window` : list/tuple, optional
        Format: [window_name, flags]
            `window_name` is the name for the output window.
            `flags` are the relevat flags that can be passed to cv2.namedWindow
        If provided, displays output in a window with given window information.
        Defaults to ["Output", cv2.WINDOW_GUI_NORMAL]

    Returns
    -------
    A Tracker object.
    """

    def __init__(
        self, detector=None, confidence=0.4, tracker='KCF',
        refresh_interval=40, video=None, width=400, window=None
            ):

        # Initialize the tracker
        if tracker.lower() == 'boosting':
            self.tracker = cv2.TrackerBoosting_create()
            print('Using Boosting tracker.')
        elif tracker.lower() == 'medianflow':
            self.tracker = cv2.TrackerMedianFlow_create()
            print('Using MedianFlow tracker.')
        else:
            try:
                self.tracker = eval(f'cv2.Tracker{tracker.upper()}_create()')
                print(f'Using {tracker.upper()} tracker.')
            except AttributeError:
                raise Exception(f"'{tracker}' is not a valid tracker name.")

        # Initialize the video stream
        if video is None:  # If no video file is provided
            print('Using webcam...\n')
            self.vs = VideoStream(src=0).start()
        elif not os.path.exists(video):
            raise Exception(
                f"The specified file '{video}' couldn't be loaded.")
        else:
            self.vs = cv2.VideoCapture(video)

        # Initialize the detector
        if detector is not None:  # If a detector is provided
            try:
                self.detector = cv2.dnn.readNetFromCaffe(*detector)
            except Exception as e:  # If the model fails to load
                print(f"The detector couldn't be initialized.")
                raise(e)
        else:
            self.detector = None

        # Sanity check for confidence
        if 0. < confidence < 1.:
            self.confidence = confidence
        else:
            raise Exception("Confidence should lie within 0 and 1.")

        # Save the output window options
        if window is None:  # If no window information is provided,
            # Create a window with the default options
            self.window = ["Output", cv2.WINDOW_GUI_NORMAL]
        else:
            self.window = window

        # Initialize other attributes
        self.initBB = None  # to store bounding box coordinates
        self.fps = None  # Initialize fps (frames per second) count
        self.frame = None  # For storing the frame to be displayed
        self.frame_copy = None  # For storing the unmodified frame
        self.interval = refresh_interval  # The refresh interval

        # Some less important attributes
        self.tracker_name = tracker
        self.using_webcam = True if video is None else False
        self.width = width  # The width to resize the frame to

    def get_BB(self):
        """
        Tracker.get_BB(self)

        Get the bounding box to be tracked.
        If a detector was provided earlier, gives the bounding box
        detected by the detector.
        Else, lets the user select a ROI by himself.
        """

        H, W = self.frame.shape[:2]  # Grab the shape of the frame

        if self.detector is not None:  # If a detector is provided

            # Preprocess frame to pass through the detector and create a blob
            blob = cv2.dnn.blobFromImage(
                cv2.resize(self.frame_copy, (300, 300)), 1.,
                (300, 300), (104.0, 177.0, 123.0)
                )
            self.detector.setInput(blob)  # Set the input image for detector

            # Get the detections.
            detections = self.detector.forward()

            if detections is not None:  # If anything is detected at all
                # The returned detections are sorted according to
                # their confidences
                # Therefore, the first element is the one we want
                newBB = detections[0, 0, 0, :].squeeze()
                newBB_confidence = newBB[2]

                # Compute (x, y) coordinates for the bounding box
                box = newBB[3:7] * np.array([W, H, W, H])
                (startX, startY, endX, endY) = box.astype("int")

                w = endX - startX  # Width
                h = endY - startY  # Height
                newBB = [startX, startY, w, h]  # The required format

                # Update the bounding box only if it has greater
                # confidence than the self.confidence threshold
                if newBB_confidence >= self.confidence:
                    self.initBB = newBB
                    # Finally, initialize the tracker
                    self.tracker.init(self.frame, tuple(self.initBB))
                else:
                    pass

            else:  # If nothing is detected
                # Set self.initBB to None, so that this function
                # will be called again and again till it finally
                # detects the object
                self.initBB = None

        else:  # If a detector is not provided

            # Put some on-screen instructions to select the ROI
            info = [
                ('ESC', 'Reselect region'),
                ('ENTER or SPACE', "Confirm selection"),
                ('Select region of interest.', ''),
            ]

            for i, (label, value) in enumerate(info):
                text = f"{label}: {value}"
                cv2.putText(
                    self.frame, text, (10, H - ((i * 20) + 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2
                )

            # Select ROI and use as our bounding box
            self.initBB = cv2.selectROI(self.window[0], self.frame,
                                        showCrosshair=False)

            self.frame = self.frame_copy  # Clear the text on the frame

            # Finally, initialize the tracker
            self.tracker.init(self.frame, tuple(self.initBB))

    def start(self):
        """
        Tracker.start(self)

        Start the detection and tracking pipeline. Generates a window
        showing the results of tracking.

        Usage
        -----
        On the generated output window, press following keys
        to perform the function listed next to them.
        +--------------------+----------------------------------------+
        |        Key         |                Function                |
        +--------------------+----------------------------------------+
        |         S          |    Re-initialize the bounding box.     |
        | Q or Esc or ALT+F4 |    Close the window and stop tracking. |
        +--------------------+----------------------------------------+

        """

        # --TESTING-- #
        # Create the output window
        try:
            window = cv2.namedWindow(*self.window)
        except Exception as e:
            print('Couldn\'t create window.')
            raise(e)
        # --TESTING-- #

        while True:
            self.frame = self.vs.read()  # Grab a frame from the video
            self.frame = \
                self.frame[1] if not self.using_webcam else self.frame

            # To reduce the processing time
            self.frame = resize(self.frame, width=self.width)
            H, W = self.frame.shape[:2]  # Height and width, needed later on
            self.frame_copy = self.frame.copy()  # Preserve an original copy

            if self.frame is None:  # Marks the end of the stream
                print('Stream has ended, exiting...')
                break

            if self.initBB is None:  # The tracker is not initialized.
                self.get_BB()  # Get the initial bounding box
                self.fps = FPS().start()  # Start recording FPS

            else:
                # Get the updated bounding box from tracker
                success, BB = self.tracker.update(self.frame)

                if success:  # If succeded in tracking
                        x, y, w, h = [int(item) for item in BB]
                        cv2.rectangle(self.frame, (x, y), (x+w, y+h),
                                      (0, 255, 0), 2)

                # Update the FPS counter
                self.fps.update()
                self.fps.stop()

                # Put some info on the frame
                info = [
                        ('Tracker', self.tracker_name),
                        ('Success', 'Yes' if success else 'No'),
                        ('FPS', f'{round(self.fps.fps(), 2)}')
                    ]

                for i, (label, value) in enumerate(info):
                    text = f"{label}: {value}"
                    cv2.putText(
                        self.frame, text, (10, H - ((i * 20) + 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
                    )

            # cv2.imshow(self.window, self.frame)
            cv2.imshow(self.window[0], self.frame)  # --TESTING-- #
            key = cv2.waitKey(1) & 0xFF  # Get the keycode for pressed key

            # If 'S' is pressed, re-initialize the bounding box
            if key == ord('s'):
                self.get_BB()  # Get a new bounding box
                self.fps = FPS().start()  # Restart the FPS counter

            elif (key == ord('q') or key == 27  # if 'Q' or 'ESC' is pressed,
                    or not cv2.getWindowProperty(  # or if the window is closed
                        self.window[0], cv2.WND_PROP_VISIBLE
                        )
                  ):

                print('Exiting...')
                break  # Stop the stream

        self.stop()  # Release the resources and cleanup

    def stop(self):
        """
        Tracker.stop(self)

        Releases resources. Destroys all OpenCV Windows and releases
        file pointer to the video (if one was given) or stops the webcam
        (otherwise).
        """
        # If we were using webcam, stop it
        if self.using_webcam:
            self.vs.stop()

        # Otherwise, release the file pointer to tje video provided
        else:
            self.vs.release()

        # Close all windows
        cv2.destroyAllWindows()


if __name__ == '__main__':

    # Please ensure the following files are present
    # ./deploy.prototxt.txt
    # ./res10_300x300_ssd_iter_140000.caffemodel

    # A test example
    tracker = Tracker(
        detector=("./deploy.prototxt.txt",
                  "./res10_300x300_ssd_iter_140000.caffemodel"),
        confidence=0.4,
        tracker='kcf',
        refresh_interval=40,
        video=None,
        width=400,
        window=None
      )
    tracker.start()
