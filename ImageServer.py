from LIBRARIES import *
from CONFIG import *

'''
Two modalities needed:
1. Continuous Stream from multiple cameras
2. Similated streams from multiple Video sources

Central function: getNextSecondFrames()
Return all the frames related to the last second of recording
Multithreading function needed


'''

#########################
#   SINGLE THREAD MOD   #
#########################

class camThread(threading.Thread):

    def __init__(self, name, camID):
        threading.Thread.__init__(self)
        self.name = name
        self.camID = camID

        self.cam = None
        self.fps = 0

        self.frameIndex = 0
        self.timeIndex = 0
        self.cameraBuffer = {}
        self.tempBuffer = []

        self.topIndex = 0
        self.maxBufferSize = MAX_BUFFER_SIZE
        self.bufferReduction = BUFFER_REDUCTION

        self.nextSecondIndex = 0

    def run(self):
        print("Starting:", self.name)

        self.cam = cv2.VideoCapture(camID)
        self.fps = cam.get(cv2.cv.CV_CAP_PROP_FPS)

        # Try to get the first frame
        if self.cam.isOpened():
            rval, frame = self.cam.read()
            self.newFrame(frame)
        else:
            rval = False

        while rval:
            rval, frame = cam.read()
            self.newFrame(frame)

            if self.frameIndex % self.fps == 0:
                self.newSecond()

    def newFrame(self, frame):
        self.tempBuffer.append(np.array(camera))
        self.frameIndex += 1

    def newSecond(self):
        self.cameraBuffer[self.timeIndex] = self.tempBuffer
        self.tempBuffer = []
        self.timeIndex += 1
        self.frameIndex = 0

        if self.timeIndex - self.topIndex > self.maxBufferSize:
            counter = self.bufferReduction
            for key in self.cameraBuffer.keys():
                del self.cameraBuffer[key]
                counter -= 1
                if counter == 0:
                    break
            self.topIndex += self.bufferReduction


    def getNextSecondFrames(self):
        nextSecond = cameraBuffer[self.nextSecondIndex]
        self.nextSecondIndex += 1
        return nextSecond
