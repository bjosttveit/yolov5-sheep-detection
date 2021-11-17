Using YOLOv5: <https://github.com/ultralytics/yolov5>
# Sheep detection with YOLOv5 🚀 🐑
Project at NTNU detecting sheep in drone images.

<img width="666" alt="bilde" src="https://user-images.githubusercontent.com/47412359/142167255-062beae0-410e-4224-8352-b6488e33b2dc.png">

## 👩🏻‍🎓 Dependencies
- Kaggle: `pip3 install kaggle` (follow official instructions for authentication)
- Weights & biases: `pip3 install wandb` (follow official instructions for authentication)

## 🧑🏽‍🔧 Install YOLOv5 into the root dir.
1. Clone the repo: `git clone https://github.com/ultralytics/yolov5.git`.
2. Install the necessary dependencies: `cd yolov5` & `pip3 install -r requirements.txt`.

## 🕵🏾‍♂️ Download the dataset
Run `./data.sh` to download the training data from kaggle.

## 👩🏼‍🏫 Download the models
Run `python3 models.py` to download the custom models trained on the data.

## 👩🏻‍🔬 Test performance
Run `./test.sh [n|s|m|l|x]` to evaluate a model on the test set. In order to test the high-res models (n6, s6, etc.) run `./test6.sh [n6|s6|m6|l6|x6]` instead. The results are saved under `tests/`. This will also save the predictions which can be used to evaluate the number of images containing sheep where at least one true positive was predicted at different confidence thresholds. To do that, run `python3 custom_recall.py [n|n6|s|s6|m|m6|l|l6|x|x6]`.

To test the inference speed, run `./speedtest.sh [n|s|m|l|x] [n_trials]`. Similarly for high-res models, run `./speedtest6.sh [n6|s6|m6|l6|x6] [n_trials]`. This will run through the test-set at a batch size of 1, measure the inference speed in milliseconds and take the average. It will run `n_trial` times and append the results to a file under `speeds/`.
