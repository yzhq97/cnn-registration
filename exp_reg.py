from __future__ import print_function
import time
import numpy as np
import Registration
from utils.utils import *
import cv2

datadir = '/media/yzhq/TOSHIBA/data/Registration/satellite/'
imgoutdir = datadir + 'out/'
#textoutdir = '/Users/yzhq/Code/MATLAB/data/RemoteSense/TEST/textout/'
name1 = '10a'
name2 = '10b'
ext = '.jpg'
IX_path = datadir + name1 + ext
IY_path = datadir + name2 + ext

IX = cv2.imread(IX_path)
IY = cv2.imread(IY_path)
# IX = cv2.resize(IX, (IX.shape[1], IX.shape[0]))
# IY = cv2.resize(IY, (IY.shape[1], IY.shape[0]))
shape = IX.shape[:2]
shape_arr = np.array(shape)
center = shape_arr / 2.0
window_width = shape[1]
window_height = shape[0]

print('initializing')
reg = Registration.CNN7()
reg.init_thres = 1.085

print('registering')
X, Y, T = reg.register(IX, IY)

print('generating warped image')
registered = tps_warp(Y, T, IY, IX.shape)
registered_name = 'registered_' + name1 + '_' + name2 + '.jpg'
cv2.imwrite(imgoutdir + registered_name, registered)
print('saved to ' + registered_name)

print('generating checkboard image')
cb = checkboard(IX, registered, 11)
cb_name = 'checkboard_' + name1 + '_' + name2 + '.jpg'
cv2.imwrite(imgoutdir + cb_name, cb)
print('saved to ' + cb_name)

while True:
  cv2.imshow('checkboard', cb)
  k = cv2.waitKey(4) & 0xFF
  if k == 27:
     break
cv2.destroyAllWindows()
del cb, IY
print()

print('please select landmark')

landmark_sensed = []
landmark_registered = []
def sensed_callback(event, j, i, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        landmark_sensed.append((i, j))
        cv2.circle(IX, (j, i), 3, (0, 0, 255), -1)
        cv2.putText(IX, str(len(landmark_sensed)), (j, i-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
def registered_callback(event, j, i, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        landmark_registered.append((i, j))
        cv2.circle(registered, (j, i), 3, (0, 0, 255), -1)
        cv2.putText(registered, str(len(landmark_registered)), (j, i-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
cv2.namedWindow('sensed', cv2.WINDOW_NORMAL) # Can be resized
cv2.resizeWindow('sensed', width=window_width, height=window_height)
cv2.setMouseCallback('sensed', sensed_callback) #Mouse callback
cv2.namedWindow('registered', cv2.WINDOW_NORMAL) # Can be resized
cv2.resizeWindow('registered', width=window_width, height=window_height)
cv2.setMouseCallback('registered', registered_callback) #Mouse callback
while True:
  cv2.imshow('sensed', IX)
  cv2.imshow('registered', registered)
  k = cv2.waitKey(4) & 0xFF
  if k == 27:
     break
cv2.destroyAllWindows()

landmark_name = 'landmark_' + name1 + '_' + name2 + '.jpg'
cv2.imwrite(imgoutdir + landmark_name, IX)
print('saved to ' + registered_name)

LMX = np.array(landmark_sensed)
LMY= np.array(landmark_registered)
N1, N2 = LMX.shape[0], LMY.shape[0]
assert N1 == N2

dist2 = np.sum((LMX-LMY)**2, axis=1)
dist = np.sqrt(dist2)
absdist = np.sum(np.abs(LMX-LMY), axis=1)

rmse = np.sqrt(np.mean(dist2))
mae = np.mean(absdist)
std = np.std(dist)
mee = np.median(dist)

print()
print('rmse %f' % rmse)
print('mae %f' % mae)
print('mee %f' %mee)
print('std %f' % std)
