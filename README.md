# cnn-registration
A image registration method using convolutional neural network features written in Python2, Tensorflow API r1.5.0.
![Alt text](https://github.com/yzhq97/cnn-registration/raw/publish/img/process_comp.jpg)

# Requirements
To install all the requirements run
```
pip install -r requirements.txt
```

# Usage
```python
import Registration
from utils.utils import *
import cv2

# load images
IX = cv2.imread(IX_path)
IY = cv2.imread(IY_path)

#initialize
reg = Registration.CNN()

#register
X, Y, Z = reg.register(IX, IY)

#generate regsitered image using TPS
registered = tps_warp(Y, Z, IY, IX.shape)
```