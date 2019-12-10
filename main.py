from BufferManager import *
from ImageServer import *
from CONFIG import *
from LIBRARIES import *

second = True

cameraThread1 = camThread("CameraTest1", 0)
cameraThread1.start()

if second:
    cameraThread2 = camThread("CameraTest2", 1)
    cameraThread2.start()

time.sleep(2)
cv2.namedWindow("cam0")
cv2.namedWindow("cam1")
while(True):
    output1 = cameraThread1.getNextSecondFrames()
    cv2.imshow("cam0", np.squeeze(output1[0,...]))
    print("cam0:", output1.shape)
    if second:
        output2 = cameraThread2.getNextSecondFrames()
        print("cam1:", output2.shape)
        cv2.imshow("cam1", np.squeeze(output2[0,...]))
    key = cv2.waitKey(20)
    time.sleep(1)
