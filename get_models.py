import os
import wandb

api = wandb.Api()

runs = [
    {"run": "run_3kpe4dbj_model:v0", "model": "RGB-n"},
    {"run": "run_233f17y6_model:v0", "model": "RGB-s"},
    {"run": "run_3p0v997m_model:v0", "model": "RGB-n6"},
    {"run": "run_1lj0oubv_model:v0", "model": "RGB-m"},
    {"run": "run_130r3xgx_model:v0", "model": "RGB-s6"},
    {"run": "run_3r3v0n9c_model:v0", "model": "RGB-l"},
    {"run": "run_39plnshp_model:v0", "model": "RGB-x"},
]

if not os.path.exists('models'):
    os.makedirs('models')

for run in runs:
    api.run(f"bjosttveit/YOLOv5/{run['run']}").file("best.pt").download(root="./models", replace=True)
    break