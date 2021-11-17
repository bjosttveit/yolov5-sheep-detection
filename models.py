import os
import wandb

api = wandb.Api()

runs = [
    {"artifact": "run_itjnuk34_model:v0", "model": "RGB-n"},
    {"artifact": "run_31qloi6u_model:v0", "model": "RGB-n6"},
    {"artifact": "run_2fgl0fsr_model:v0", "model": "RGB-s"},
    {"artifact": "run_83zyol33_model:v0", "model": "RGB-s6"},
    {"artifact": "run_486j5y0q_model:v0", "model": "RGB-m"},
    {"artifact": "run_3minhw2k_model:v0", "model": "RGB-m6"},
    {"artifact": "run_2octyttq_model:v0", "model": "RGB-l"},
    {"artifact": "run_3kidhd8s_model:v0", "model": "RGB-l6"},
    {"artifact": "run_38emofsu_model:v0", "model": "RGB-x"},
    #{"artifact": "run_[ID]_model:v0", "model": "RGB-x6"},
]

if not os.path.exists('models'):
    os.makedirs('models')

for run in runs:
    api.artifact(f"YOLOv5/{run['artifact']}").download(root="./models")
    os.rename("models/best.pt", f"models/{run['model']}.pt")
