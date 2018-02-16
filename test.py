import numpy as np
from utils.utils import *

x = np.array([
    [1,1], [0, 1], [1, 1]
], dtype='float32')

y = np.array([
    [2,0], [-1, 2]
], dtype='float32')

print(gaussian_radial_basis(x, 2.0))