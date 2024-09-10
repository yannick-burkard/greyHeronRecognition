## Grey Heron Detection

This repository contains the `detection.py` script that applies a given object detection model in YOLOv5 format on a specified dataset.
By default, a .txt file containing labels (class, boundix box, confidence) in YOLOv5 format is saved for every image in which objects are detected.
It is also possible to save labelled images, cropped detections, as well as a csv file containing results.
The function takes in as arguments a dictionary containing the following detection configurations:
- `'parent_dir'`: this is the parent directory corresponing to the greyHeronRecognition repository (e.g. `'/cluster/home/username/greyHeronRecognition'`)
- `'yolo_path'`: path of yolov5 repository (relative to parent directory)
- `'data_path'`: can be string or list containing directory(ies) and / or file paths (both relative to the parent directory) of the images to be detected (e.g. `['data/dataset/dataSDSC/SBU3','data/dataset/dataSDSC/SBU2','data/dataset/dataSDSC/NEN/2017_NEN1_02090354.JPG']` 
- `'model_path'`: path of the YOLOv5 model (e.g. `'models/detection/md_zenodo_v5b0.0.0.pt'` for Megadetector, `'greyHeronDetection/framework_pwl/runs/train/20240615_204402/results/weights/last.pt'` for fine-tuned model)
- `'iou_tsh'`: IoU threshold for non-max supression, default for yolov5 is 0.45
- `'imgsz'`: input image resolution
- `'n_gpus'`: number of gpus being used
- `'save_dir'`: path of directory where results are saved (relative to parent directory)
- `'save_im'`: boolean to save labelled images or not
- `'save_crop'`: boolean to save cropped detections or not (yolov5 argument)
- `'save_csv'`: boolean to save results to csv file or not (yolov5 argument)
- `'conf_tsh'`: specifies confidence threshold; if a fine-tuned model is used, this variable can further be set to
  - `'get_config'`: extract confidence threshold saved in training configurations
  - `'get_stats'`: extract confidence thereshold yielding maximum F1 score during training

To use this script, specify the configurations in a dictionary `config_detect`, call `detect_config(config_detect)` below and run the python script.
