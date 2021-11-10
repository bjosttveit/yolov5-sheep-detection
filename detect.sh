#!/bin/sh

cd yolov5
python3 detect.py --source ../datasets/rgb/images --weights ../models/RGB-m.pt --nosave
