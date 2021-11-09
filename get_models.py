import os
import wandb

api = wandb.Api()

runs = [
    {"artifact": "run_3kpe4dbj_model:v0", "model": "RGB-n"},
    {"run": "233f17y6", "model": "RGB-s"},
    {"run": "3p0v997m", "model": "RGB-n6"},
    {"run": "1lj0oubv", "model": "RGB-m"},
    {"run": "130r3xgx", "model": "RGB-s6"},
    {"run": "3r3v0n9c", "model": "RGB-l"},
    {"run": "39plnshp", "model": "RGB-x"},
]

if not os.path.exists('models'):
    os.makedirs('models')

api.artifact(f"YOLOv5/{runs[0]['artifact']}").download(root="./models")

#for run in runs:
#    api.run(f"bjosttveit/YOLOv5/{run['run']}").file("best.pt").download(root="./models", replace=True)
#    break