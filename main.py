from BufferManager import *
from ImageServer import *
from CONFIG import *
from LIBRARIES import *


cameraThread = camThread("CameraTest", 1)

while(True):
    print(cameraThread.getNextSecondFrames().shape)
    time.sleep(1)
