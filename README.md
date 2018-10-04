# HeadPose
Dataset Link : https://drive.google.com/file/d/0B7OEHD3T4eCkVGs0TkhUWFN6N1k/view?usp=sharing 

## Requirements 
  Module                |   Version
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
* Open Training.ipyb 
* Change "DATASET_PATH" to path where dataset is kept
* Run other cells to complete training

## Testing
* Run run.py

## Using Object Tracker script
* The script can be run from the command line with or without any arguments.
```bash
usage: object tracker.py [-h] [-v VIDEO] [-t TRACKER]

optional arguments:
  -h, --help            show this help message and exit
  -v VIDEO, --video VIDEO
                        Path to the input video file.
  -t TRACKER, --tracker TRACKER
                        Name of the tracker to use.

```
- If no `VIDEO` is provided, the script uses webcam by default.
- If no `TRACKER` is provied, the script uses **KCF** by default. 
- Available trackers in `OpenCV` 3.4+:
<table><kbd>
<tr><td>Boosting</td><td>MIL</td><td>KCF</td><td>MedianFlow</td><td>MOSSE</td><td>CSRT</td><td>TLD</td></tr>
</kbd></table>

- After the script runs, select the region on the frame, and it will start tracking and will show some statistics on the screen.
