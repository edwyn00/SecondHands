from LIBRARIES import *
from CONFIG import *

'''
The class in this file manages the frames that arrives from the function getLastSecondFrames().
This call can be done from the ImageServer of this same project or from the ImagePacker developed by Ocado team.

This class save in a buffer all the frame received and prepare them for the various components of the projects:
1. Mask RCNN needs a frame per stream (the last of the previous second).
2. AFN needs a buffer of 6 frames samples from each of the last 5 seconds.
2. ActivityGAN needs few frames from the last second.

Possible work: integrate the preprocessing steps on the images in this class
'''
