#!/bin/sh

# Uncomment when running on IDUN
#module purge
#module load Python/3.8.6-GCCcore-10.2.0

mkdir -p speeds

cd yolov5

for i in `seq 1 $2`
do
    echo "Running speedtest $i of $2..."
    python3 val.py --data ../datasets/rgb.yaml --weights ../models/RGB-${1}.pt --project ../tests --name RGB-${1} --exist-ok --task=speed --batch-size=1 2>&1 | sed -En 's/.*pre-process, (.*)ms inference.*/\1/p' >> ../speeds/RGB-${1}.txt
done

echo "Done!"
