# Face Recognition App Demo with QT5

<p align="center">
  <img alt="Light" src="https://files.seeedstudio.com/wiki/reTerminal_ML/face_recognition.gif" width="100%">
</p>

Sample application for Face Recognition on Seeed Studio reTerminal made with QT5 and PySide2. Will work on Linux PC and other SBCs as well, provided necessary requirements are met.

## Installation

reTerminal Raspberry Pi OS image comes with QT5 and PySide components pre-installed. 

Install additional dependencies:

```bash
pip3 install -r requirements.txt
```

[Install TensorFlow Lite Interpreter package](https://wiki.seeedstudio.com/reTerminal_ML_TFLite/#tensorflow-lite-runtime-package-installation)  - for faster inference install XNNPACK supported version.

[Download the models](https://files.seeedstudio.com/ml/face_rec_models/face_rec_models.zip) and place them inside of face_rec_models directory.

**Note** Feature extraction model originally is from this [GitHub repository](https://github.com/zye1996/Mobilefacenet-TF2-coral_tpu), is has accuracy of 99.3% on LFW dataset. The author didn't leave a LICENSE file in the repository, so the legal status of using pre-trained models provided for commercial purposes is undefined. You can use provided models for evaluation and development, and we're working now on fully open source models to replace them in near future.

## Usage

Run 
```bash
DISPLAY=:0 python3 main.py
```
from the project main folder. Use the menu to record and delete new faces to the database.

## TO-DO

- [ ] Pre-trained models published under MIT License
- [ ] GPIO Control example for reTerminal
- [ ] MQTT example
- [ ] Support for Raspberry Pi camera