#!/bin/sh

# Uncomment when running on IDUN
#module purge
#module load Python/3.8.6-GCCcore-10.2.0

cd yolov5
python3 val.py --data ../datasets/rgb.yaml --weights ../models/RGB-n.pt --task speed
