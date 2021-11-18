#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=ie-idi
#SBATCH --time=16:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:V100:2
#SBATCH --cpus-per-task=24
#SBATCH --mem=16G
#SBATCH --job-name="YOLO_RGB_n"
#SBATCH --output=yolo-rgb-n.out
#SBATCH --mail-user=bjornar.osttveit@ntnu.no
#SBATCH --mail-type=ALL


cd ${SLURM_SUBMIT_DIR}/yolov5

module purge
module load Python/3.8.6-GCCcore-10.2.0

pip3 install -r requirements.txt --user -q
pip3 install wandb --user -q

#python3 train.py --batch 128 --device 0 --epochs 1000 --data ../datasets/rgb.yaml --weights yolov5n.pt
python3 -m torch.distributed.launch --nproc_per_node=2 train.py --batch 256 --device 0,1 --workers 16 --epochs 1000 --data ../datasets/rgb.yaml --weights yolov5n.pt
