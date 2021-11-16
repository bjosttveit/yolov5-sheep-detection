#!/bin/sh

# Uncomment when running on IDUN
#module purge
#module load Python/3.8.6-GCCcore-10.2.0

mkdir -p tests

cd yolov5
python3 val.py --data ../datasets/rgb.yaml --weights ../models/RGB-${1}.pt --project ../tests --name RGB-${1} --exist-ok --task=test --save-txt --save-conf --img 1280
