from BufferManager import *
from ImageServer import *
from CONFIG import *
from LIBRARIES import *


testBufferManager = True


def bufferManagerTesting():
    bm = BufferManager("Buffer Manager")
    bm.start()
 

if testBufferManager:
    bufferManagerTesting()
