from imutils.video import VideoStream, FPS
import imutils
# import numpy as np  # Use if you want to flip the stream horizontally.
import argparse
import cv2
import sys
import os

# Fallback tracker if no tracker is provided
default_tracker = 'KCF'  # Available: [CSRT, MIL, Boosting, MedianFlow, MOSSE]
initBB = None  # It will store bounding box coordinates
fps = None  # Initialize fps (frames per second) count

ap = argparse.ArgumentParser()
ap.add_argument('-v', '--video', type=str, required=False,
                help='Path to the input video file.', default=None)
ap.add_argument('-t', '--tracker', type=str, required=False,
                help='Name of the tracker to use.', default=default_tracker)

args = vars(ap.parse_args())

# Choosing the tracker to be used
if args['tracker'].lower() == 'boosting':
    tracker = cv2.TrackerBoosting_create()
    print('Using Boosting tracker.')
elif args['tracker'].lower() == 'medianflow':
    tracker = cv2.TrackerMedianFlow_create()
    print('Using MedianFlow tracker.')
else:
    try:
        tracker = eval(f'cv2.Tracker{args["tracker"].upper()}_create()')
        print(f'Using {args["tracker"].upper()} tracker.')
    except AttributeError:
        print('Invalid tracker name provided.')
        sys.exit(0)

# Initialize video stream
if args['video'] is None or not os.path.exists(args['video']):
    print('Using webcam...\n'
          '[Either because no video file was provided or'
          ' because the file specified couldn\'t be found...]')
    vs = VideoStream(src=0).start()
else:
    vs = cv2.VideoCapture(args['video'])


# A helper function for testing.
def _selectRegion(window, tracker, frame_copy):
    """
    Allows the user to select a ROI for tracking.
    Only for testing purposes.
    """

    # Put instructions on the screen to select ROI
    info = [
        ('ESC', 'Reselect region'),
        ('ENTER or SPACE', "Confirm selection"),
        ('Select region of interest.', ''),
    ]
    for i, (label, value) in enumerate(info):
        text = f"{label}: {value}"
        cv2.putText(
            frame_copy, text, (10, H - ((i * 20) + 20)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2
        )

    # Use the selected ROI and use as our bounding box
    initBB = cv2.selectROI(window, frame_copy, showCrosshair=False)
    return frame_copy, initBB


while True:
    frame = vs.read()  # Grab a frame from the video
    frame = frame[1] if args['video'] is not None else frame
    # frame = np.fliplr(frame)  # Flip the image (Causes a drop of 1-2 FPS)

    if frame is None:  # Signifies end of the stream
        print('Stream has ended, exiting...')
        break

    frame = imutils.resize(frame, width=400)  # To reduce processing time
    H, W = frame.shape[:2]
    frame_copy = frame.copy()

    if initBB is None:
        frame, initBB = _selectRegion("Output", tracker, frame_copy)
        tracker.init(frame, initBB)  # Initialize the tracker
        fps = FPS().start()  # Start recording FPS
    else:
        success, BB = tracker.update(frame)  # Get the updated box from tracker

        if success:  # If succeded in tracking
                x, y, w, h = [int(item) for item in BB]
                cv2.rectangle(frame, (x, y), (x+w, y+h),
                              (0, 255, 0), 2)

        # Update the FPS counter
        fps.update()
        fps.stop()

        # Put some info on the frame
        info = [
            ('Tracker', args['tracker']),
            ('Success', 'Yes' if success else 'No'),
            ('FPS', f'{fps.fps()}')
        ]
        for i, (label, value) in enumerate(info):
            text = f"{label}: {value}"
            cv2.putText(
                frame, text, (10, H - ((i * 20) + 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
            )

    cv2.imshow('Output', frame)
    key = cv2.waitKey(1) & 0xFF

    # If 'S' is pressed, let the user select a bounding box to track.
    if key == ord('s'):
        frame, initBB = _selectRegion("Output", tracker, frame_copy)
        tracker.init(frame, initBB)
        fps = FPS().start()

    # Stop video if 'Q' or 'ESC' is pressed, or if the window is closed.
    elif (key == ord('q') or key == 27
            or not cv2.getWindowProperty("Output", cv2.WND_PROP_VISIBLE)):
        print('Exiting...')
        break

# If we were using webcam, release it
if args['video'] is None:
    vs.stop()
# Otherwise, release the file pointer
else:
    vs.release()

# Close all windows
cv2.destroyAllWindows()
