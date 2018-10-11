# HeadPose
Dataset Link : https://drive.google.com/file/d/0B7OEHD3T4eCkVGs0TkhUWFN6N1k/view?usp=sharing

## Requirements
Module         |   Version
  :----------------------:|:--------------:
  `scipy`                 |      1.1.0
  `numpy`                 |     1.14.3
  `keras`                 |      2.2.2
  `keras-applications`    |      1.0.4                    
  `imutils`               |      0.5.1
  `opencv-python`         |      3.4 or greater
  `opencv-contrib-python` |     3.4 or greater

Scripts written and tested in Python v3.6.5

## How to train
* Open `Training.ipynb`
* Change "DATASET_PATH" to path where dataset is kept
* Run other cells to complete training

## Testing
* Run run.py

## The tracker module
### Docstring
```bash
  Tracker(
      self, detector=None, confidence=0.4, tracker='KCF',
      refresh_interval=20, video=None, width=400, window=None
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
      bounding box. Defaults to 5. Ignored if a `detector` is not provided.
      Helps the tracker maintain the correct bounding box.
      Set equal to 0 to disable this feature.

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
```
### Usage Example
```python
from tracker import Tracker

# Create a tracker object to track faces
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

# Start tracking
tracker.start()
```
### Testing
The above mentioned example is provided in the `tracker.py` itself. Just run the script to see the results.
