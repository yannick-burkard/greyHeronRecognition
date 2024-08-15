#Grey Heron Detection

This repository contains the `detection.py` script that applies a given object detection model in YOLOv5 format on a specified dataset.
The funtion takes in as arguments a dictionary containing the following detection configurations:
1. `'parent_dir'`: this is the parent directory corresponing to the greyHeronRecognition repository (ex.: `'/cluster/home/username/greyHeronRecognition`)
2. `''yolo_path'`: path of yolov5 repository (relative to parent directory)
3. `'data_path'`: can be string or list containing directory(ies) and / or file paths (both relative to the parent directory) of the images to be detected (ex. `['data/SBU3','data/SBU2','data/NEN/2017_NEN1_02090354.JPG']`)
