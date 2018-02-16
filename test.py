import numpy as np
import cv2
import matplotlib.pyplot as plt

datadir = '../data/RemoteSense/ANGLE/68/'
#datadir = '/Users/yzhq/Code/MATLAB/data/RemoteSense/ANGLE/68/'
IX_path = datadir + '2.jpg'
IY_path = datadir + '5.jpg'

IX = cv2.imread(IX_path)
IX_gray = cv2.cvtColor(IX, cv2.COLOR_BGR2GRAY)
SIFT = cv2.xfeatures2d.SIFT_create()
XS0, DXS0 = SIFT.detectAndCompute(IX_gray, None)
XS, YS, DXS, DYS = [], [], [], []
for i in range(len(XS0)):
    if XS0[i].response > 0.04:
        XS.append(XS0[i].pt)
        DXS.append(DXS0[i])
XS, YS, DXS, DYS = np.array(XS), np.array(YS), np.array(DXS), np.array(DYS)

im = plt.imread(IX_path)
#plt.gca().invert_yaxis()
plt.imshow(im)
plt.scatter(XS[:, 0], XS[:, 1])
plt.show()