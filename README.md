# Track_Progress

Code is written in Python 3. This file uses a video to track the progress of concreting at a construction site.

## Installation


Use the package manager [pip](https://pip.pypa.io/en/stable/) to install opencv and numpy.

```bash
pip install opencv-python
pip install numpy
```

## Usage

Run
``` bash
python Tracking.py
```

 
It will show you the first frame of the video where you can select the Region of Interest (ROI) (The region in which you want to track changes).

If you want to get the basic understanding of how the code proceeds: 
You can select region by selecting/marking one point at a time and generate a close loop which can be used as ROI. You can select points by pressing left click and cancel last selected point by pressing right click. Once you have marked/selected ROI, click middle mouse button to see the ROI in a separate window. If you are satisfied with your ROI, click "s" to save ROI and it's mask. Then to exit ROI selection, press "Esc" key. Same instruction will be shown in the terminal as well. 

Then it will show you the changes in that region and the progress in that region.
This code is specifically written to track the changes in concrete flooring. If you want to use it for other changes, you can simply change the threshold value. To get the exact value of concrete, I have used HSV Trackbar `Trackbar.py`. Using this, you will able to find the threshold value as per your need.
