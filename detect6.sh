#!/bin/sh

# Uncomment when running on IDUN
#module purge
#module load Python/3.8.6-GCCcore-10.2.0

cd yolov5
python3 detect.py --source ../datasets/rgb/images --weights ../models/RGB-${1}.pt --nosave --img 1280
