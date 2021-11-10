#!/bin/sh

# Uncomment when running on IDUN
#module purge
#module load Python/3.8.6-GCCcore-10.2.0

kaggle datasets download -d bjosttveit/sheep-uav-yolo
rm -rf datasets/ir datasets/rgb
unzip -o sheep-uav-yolo.zip -d datasets
rm sheep-uav-yolo.zip

python3 ./train_test_split.py
