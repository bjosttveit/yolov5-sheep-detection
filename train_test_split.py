import sys
sys.path.insert(0, './yolov5')

from utils.datasets import *

autosplit('datasets/ir/images', (0.7, 0.15, 0.15))
autosplit('datasets/rgb/images', (0.7, 0.15, 0.15))
