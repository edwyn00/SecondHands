from LIBRARIES  import *
from CONFIG     import *


class ImageServer(threading.Thread):

    def __init__(self, name, n_cams, event):
        threading.Thread.__init__(self)
        self.name   = name
        self.n_cams = n_cams
        self.event  = event

        self.singleCamEvents    = []
        self.singleCamNames     = []
        self.singleCamIds       = []
        self.singleCamThread    = []

        self.allFrames = []

        for i in range(self.n_cams):
            self.singleCamEvents.append(threading.Event())
            self.singleCamNames.append(("cam"+str(i)))
            self.singleCamIds.append(i)
            camT = camThread(  self.singleCamNames[i],\
                                    self.singleCamIds[i]  ,\
                                    self.singleCamEvents[i])
            self.singleCamThread.append(camT)




    def run(self):
        for i in range(self.n_cams):
            print("Starting:", self.singleCamNames[i])
            self.singleCamThread[i].start()
        time.sleep(3)

        while(True):
            while all(singleEvent.isSet() for singleEvent in self.singleCamEvents):
                self.allFrames = []
                for i in range(self.n_cams):
                    frames = self.singleCamThread[i].getNextSecondFrames()
                    self.singleCamEvents[i].clear()
                    self.allFrames.append(frames)
                self.allFrames = np.stack(self.allFrames, axis=0)
                print("Sending shape:", self.allFrames.shape)
                self.event.set()

    def getNextSecondFrames(self):
        return self.allFrames
