## Grey Heron Detection

This repository contains the `detection.py` script that applies a given object detection model in YOLOv5 format on a specified dataset.
By default, a .txt file containing labels (class, boundix box, confidence) in YOLOv5 format is saved for every image in which objects are detected.
It is also possible to save labelled images, as well as a csv file containing results.
The funtion takes in as arguments a dictionary containing the following detection configurations:
1. `'parent_dir'`: this is the parent directory corresponing to the greyHeronRecognition repository (e.g. `'/cluster/home/username/greyHeronRecognition'`)
2. `'yolo_path'`: path of yolov5 repository (relative to parent directory)
3. `'data_path'`: can be string or list containing directory(ies) and / or file paths (both relative to the parent directory) of the images to be detected (e.g. `['data/SBU3','data/SBU2','data/NEN/2017_NEN1_02090354.JPG'] 
4. `'model_path'`: path of the YOLOv5 model (e.g. `'greyHeronDetection/saved_models/md_zenodo_v5b0.0.0.pt'` for Megadetector, `'greyHeronDetection/framework_pwl/runs/train/20240615_204402/results/weights/last.pt'` for fine-tuned model)
5. `'iou_tsh'`: IoU threshold for non-max supression, default by yolov5 is 0.45
6. `'imgsz'`: input image resolution
7. `'n_gpus'`: number of gpus being used
8. 'save_dir': path of directory where results are saved (relative to parent directory)
9. `'save_im'`: boolean to save labelled images or not
10. `'save_crop'`: boolean to save cropped detections or not (yolov5 argument)
11. `'save_csv'`: save results to csv file or not (yolov5 argument)
12. `'conf_tsh'`: specifies confidence threshold; if a fine-tuned model is used, this variable can further be set to
                'get_config': extract confidence threshold saved in training configurations
                'get_stats': extract confidence thereshold yielding maximum F1 score during training
To use this script, specify the configurations in a dictionary `config_detect`, run `detect_config(config_detect)` below and execute the python script.
