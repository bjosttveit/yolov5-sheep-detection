import os
import wandb

api = wandb.Api()

runs = [
    {"artifact": "run_3kpe4dbj_model:v0", "model": "RGB-n"},
    {"artifact": "run_233f17y6_model:v0", "model": "RGB-s"},
    {"artifact": "run_3p0v997m_model:v0", "model": "RGB-n6"},
    {"artifact": "run_1lj0oubv_model:v0", "model": "RGB-m"},
    {"artifact": "run_130r3xgx_model:v0", "model": "RGB-s6"},
    {"artifact": "run_3r3v0n9c_model:v0", "model": "RGB-l"},
    {"artifact": "run_39plnshp_model:v0", "model": "RGB-x"},
]

if not os.path.exists('models'):
    os.makedirs('models')

for run in runs:
    api.artifact(f"YOLOv5/{run['artifact']}").download(root="./models")
    os.rename("models/best.pt", f"models/{run['model']}.pt")
