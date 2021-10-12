#!/bin/sh

kaggle datasets download bjosttveit/sheep-uav-yolo
rm -rf datasets/ir datasets/rgb
unzip -o sheep-uav-yolo.zip -d datasets
rm sheep-uav-yolo.zip

module purge
module load Python/3.8.6-GCCcore-10.2.0
python3 ./train_test_split.py
