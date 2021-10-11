#!/bin/sh

kaggle datasets download bjosttveit/sheep-uav-yolo
mkdir -p datasets
unzip sheep-uav-yolo.zip -d datasets
rm sheep-uav-yolo.zip
