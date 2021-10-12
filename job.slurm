#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=ie-idi
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16000
#SBATCH --job-name="YOLO_TEST"
#SBATCH --output=yolo.out
#SBATCH --mail-user=bjornar.osttveit@ntnu.no
#SBATCH --mail-type=ALL


cd ${SLURM_SUBMIT_DIR}/yolov5

module purge
module load Python/3.8.6-GCCcore-10.2.0

pip3 install -r requirements.txt --user

python3 train.py --img 640 --batch 64 --device 0 --epochs 3 --data ../datasets/ir.yaml --weights yolov5s.pt
