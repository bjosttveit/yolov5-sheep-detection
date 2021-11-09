import os
import wandb

api = wandb.Api()

runs = [
    {"run": "3kpe4dbj", "model": "RGB-n"},
    {"run": "233f17y6", "model": "RGB-s"},
    {"run": "3p0v997m", "model": "RGB-n6"},
    {"run": "1lj0oubv", "model": "RGB-m"},
    {"run": "130r3xgx", "model": "RGB-s6"},
    {"run": "3r3v0n9c", "model": "RGB-l"},
    {"run": "39plnshp", "model": "RGB-x"},
]

if not os.path.exists('models'):
    os.makedirs('models')

files = api.run(f"bjosttveit/YOLOv5/{runs[0]['run']}").files()
for f in files:
    print(f)

#for run in runs:
#    api.run(f"bjosttveit/YOLOv5/{run['run']}").file("best.pt").download(root="./models", replace=True)
#    break