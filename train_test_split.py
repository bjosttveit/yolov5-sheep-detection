import sys
sys.path.insert(0, './yolov5')

from utils.datasets import *

autosplit('datasets/ir/images', (0.8, 0.2, 0.0))
autosplit('datasets/rgb/images', (0.8, 0.2, 0.0))
