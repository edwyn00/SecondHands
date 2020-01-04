from LIBRARIES  import *
from CONFIG     import *

'''
The class in this file manages the frames that arrives from the function getLastSecondFrames().
This call can be done from the ImageServer of this same project or from the ImagePacker developed by Ocado team.

This class save in a buffer all the frame received and prepare them for the various components of the projects:
1. Mask RCNN needs a frame per stream (the last of the previous second).
2. AFN needs a buffer of 6 frames samples from each of the last 5 seconds.
3. ActivityGAN needs few frames from the last second.

Possible work: integrate the preprocessing steps on the images in this class
'''

class BufferManager(threading.Thread):
    def __init__(self, name):
        threading.Thread.__init__(self)
        self.name = name

        self.fps = 0

        self.timeIndex = 0          #Index w.r.t. Image Server
        self.currentCallIndex = 0   #Index w.r.t. Help System Request
        self.topIndex = 0           #Index w.r.t. first elem of buffer

        self.buffer = []

        self.maxBufferSize = MAX_BUFFER_SIZE
        self.bufferReduction = BUFFER_REDUCTION

        self.nextSecondIndex = 0
        self.newSecondEvent = threading.Event()
        self.ImageServer = ImageServer("Cams", 2, self.newSecondEvent)

    def run(self):
        print("Starting:", self.name)
        self.ImageServer.start()
        time.sleep(3)
        start = time.time()
        while(True):
            while self.newSecondEvent.isSet():
                frames = self.ImageServer.getNextSecondFrames()
                self.newSecond(frames)
                self.newSecondEvent.clear()
                print("Received shape:", frames.shape)
                #print("Required time:", time.time()-start)
                #start = time.time()


    def newSecond(self, frames):
        self.buffer.append(frames)
        self.timeIndex += 1

        if self.timeIndex - self.topIndex > self.maxBufferSize:
            for index in range(self.bufferReduction):
                self.buffer.pop(index)

            self.topIndex += self.bufferReduction

    def getLastSecondFrames(self):
        if self.currentCallIndex > self.timeIndex:
            print("Required batch of frames is not available yet")
            return None
        if self.currentCallIndex < self.topIndex:
            print("Your batch calls are located too far in the past")
            print("You will be aligned with the oldest batch avaliable")
            print("This batch has the time index:", slf.timeIndex)
            self.currentCallIndex = self.topIndex
        lastSecondFrames = self.buffer[self.currentCallIndex]
        self.currentCallIndex += 1
        return lastSecondFrames
