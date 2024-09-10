Welcome to the `greyHeronRecognition` repository! This pipeline has been developed to train, evaluate and apply computer vision models to recognize the grey heron (*Ardea cinerea*) in camera trap data. There are four main directories constituting the repository: `greyHeronClassification` and `greyHeronDetection` contain the scripts to train and evaluate classification and object detection models, whereas `models` contains the (un-)trained models, and `data` is comprised of csv files listing data information, bounding box labels for detection and the camera trap dataset used in this study. We provide an additional directory `greyHeronDetection/detection` where one can perform detections on unseen data.

We now provide further details for each directory.

## `data`

Here we have stored various types of data used for training and evaluation stages. `data/csv_files` contains multiple csv files listing useful information about image samples (e.g. path, label, time stamp, corrupted, infrared, mode). File `dataSDSC_full.csv` lists all data, whereas other files are comprised of training, validation and test data obtained from the first, chronological (no suffix in name) and second, season (suffix `_split2` in name) splits. Information on each column can be found in `info_dataSDSC.txt`. Additionally, subdirectory `data/dataset` contains all image files in JPG format, and `data/detection_labels` includes bounding box labels for each positive in YOLOv5 format used for object detection.

## `models`

Here we have saved trained and untrained models that can be used in future applications. Subdirectory `models/classification` contains trained classifiers (MobileNetV2) and `models/detection` trained object detectors (YOLOv5x6)  along with the zero-shot Megadetector. Further details can be found in files `model_infos.txt` located in each subdirectory.

## `greyHeronClassification`

This part is dedicated to the training and evaluation of classifiers. 

For training and results visualisation, one run the batch script train_job.sh, which will sequentially run scripts `trainAndAnalyze.py`, `train.py` (model training and validation) and `analysis/plotLearningCurves.py` (plotting and visualisation). Arguments are as follows:
- `n_epochs`: number of training epochs
- `batch_size`: batch size
- `learning_rate`: learning rate
- `weight_decay`: weight decay
- `dropout_prob`: dropout probability (default for MobileNetV2 is 0.2)
- `seed`: torch seed
- `model_name`: model name (for MobileNetV2 use `'mobilenet’`)
- `which_weights`: unfrozen weights from last (`'last'`) or all (`'all'`) layers
- `n_last_im`: number of last images for background removal; if `'none'`, then original images are taken
- `day_night`: day- (`'day`') or night-time (`'night’`) images
- `im_size`: input resolution size (e.g. `'896’`)
- `augs`: data augmentations (e.g. `'colorjitter'`); multiple augmentations need to be separated with comma without spaces in between
- `resample_trn`: resample method for training; options are ` 'none'`,  `'undersample'`, `oversample_smote`, `'oversample_naive'`, `'log_oversample'`, `'log_oversample_2'`, `'no_resample'`
- `n_cams_regroup`: number of regrouped cameras during log oversampling
- `ls_cams`: filtered cameras; optionare are `'SBU4’` or `'all’`
- `val_full`: evaluate model on full validation set after training; options are `0` (false) or `1` (true)
- `trn_val`: train with both training and validation data merged; options are `0` (false) or `1` (true)
- `which_val`: which dataset to use for validation; options are `'val'`, `'tst'` or `'none'`
- `split`: splitting method used; options are `'chronological'` (first) or `'seasonal'` (second)

Each training procedure is identified with a time stamp `time_stamp` (e.g. `20240904_122224`) and the following outputs are generated:
- subdirectory `analysis/output/time_stamp` containing 
  - `configurations.txt`: lists training configurations
  - `data_info.txt`: contains data information
  - `ls_trn.csv` and `ls_val.csv`: lists train and validation samples paths used
  - metric curve plots in `.png` format
- subdirectory `logs/checkpoints` with model weights and metrics for the best (maximum F1-score) and last model saved under `best_model_weightsAndMetrics_time_stamp.pth` and `last_model_weightsAndMetrics_time_stamp.pth`, respectively
- subdirectory `logs/metrics` containing baseline and and model metrics of all epochs saved under `model_metrics_baseline_time_stamp.pth` and `model_metrics_allEps_time_stamp.pth`, respectively

For model evaluation, run script `evaluate_job.sh`, which executes the script `evaluate.py`. Arguments are directly specified in script `evaluate.py` with a dictionary containing specifying the following configurations:
- `’parent_dir’`: parent directory (e.g. `’/cluster/project/eawag/p05001/repos/greyHeronRecognition/'`)
- `’time_stamp’`: time stamp of training job originally
- `’num_classes’`: number of classes (for our purposes this is fixed to 2)
- `’batch_size`: batch size
- `’which_weights’`: unfrozen weights from last (`'last'`) or all (`'all'`) layers
- `’pretrained_network’`: pretrained model (for MobileNetV2 use `'mobilenet’`)
- `’n_last_im’`: number of last images for background removal; if `'none'`, then original images are taken
- `’day_night’`: day- (`'day`') or night-time (`'night’`) images
- `’im_size’`: input resolution size (e.g. `'896’`)
- `’resample’`: resample method applied to dataset; options are ` 'none'`,  `'’undersample'`, `oversample_smote`, `'oversample_naive'`, `'log_oversample'`, `'log_oversample_2'`, - `'no_resample'`
- `’ls_cams_filt’`: filtered cameras; optionare are `'SBU4’` or `'all’`
- `’split’`: splitting method used; options are `'chronological'` (first) or `'seasonal'` (second)
- `’num_workers’`: number of workers used for data loading
- `’best_last’`: load best (`'best'`) or last (`'last'`) model during training
- `’which_set’`: training ('trn'), validation ('val'), both ('trn_val') or test ('tst')

Here each evaluation process is also characterized by an individual `time_stamp` (different that the one for training), and outputs generated  are:
- subdirectory `analysis/output_eval/time_stamp` containing:
  - `configurations_eval.txt`: lists configurations for evaluation
  - `confusion_matrix_norm.png`: plotted normalized confusion matrix
  - `data_info.txt`: data information
  - `data_info.txt`: complete lists of evaluation metrics

## `greyHeronDetection`

This subdirectory is dedicated to the training, evaluation and new detections of object detection models. 

It includes a `greyHeronDetection/yolov5` repository, which has been fetched from <https://github.com/ultralytics/yolov5> and partially adapted to be used in our case study. Extensively modified scripts contain a `_mod` suffix (e.g. `train_mod.py`). Another subdirectory `framework_pwl` has scripts to train and evaluate object detection architectures. 

For model training and result visualisation, run script `train_job.sh`, which in turn calls script `train_wDataResample.py` (training and validation) and `analysis/plotLearningCurves` (plotting and visualisation).  Arguments are directly specified in `train_wDataResample.py`, and are as follows:
- `'lr0'`: initial learning rate (YOLOv5 hyperparameter)
- `'lrf'`: final learning rate (YOLOv5 hyperparameter)
- `'warmup_epochs'`: number of warmup epochs (YOLOv5 hyperparameter)
- `'time_stamp'`: time stamp of training job
- `'parent_dir'`: parent directory (e.g. `'/cluster/project/eawag/p05001/repos/greyHeronRecognition/'’`)
- `'split'`: first (`'chronological'`) or second (`'seasonal'`) split used
- `'day_night'`: day or night-time images, or both
- `'n_last_im'`: number of last images for background removal; if `'none'`, then original images are taken
- `'which_val'`: which dataset to use for validation (`'trn'`, `'val'`, `'trn_val'`, `'tst'`)
-  `'conf_tsh_fixed'`: fixed confidence threshold throughout training and validation; if set to `0`, then confidence threshold is adjusted after every epoch yielding max F1-score
- `'trn_val'`: train with both training and validation data merged
- `'resample_trn'`: resample method for training
- `'n_cams_regroup’`: number of regrouped cameras during log oversampling
- `'ls_cams'`: list of filtered cameras
- `'epochs’`: number of training epochs
- `'batch_size'`: batch size
- `'weight_decay'`: weight decay for learning step
- `'imgsz'`: image resolution
- `'optimizer'`: specifies optimizer used for learning step `('SGD'`, `'Adam'`,` 'AdamW'`)
- `'freeze'`: number of layers to freeze in YOLOv5 model
- `'model'`: name of pretrained model
- `'mosaic_prob'`: mosaic probability (YOLOv5 hyperparamter)
- `'val_dataset'`: evaluate trained model on full validation set after training
- `'val_megadetector'`: evaluate megadetector on validation set
- `'reduced_dataset'`: train and evaluate on reduced dataset, if set to int n, takes n first elements before resampling, useful to make quick checks
- `'workers'`: number of workers used for data loading 
- `'n_gpus'`: number of gpus used
- `'seed'`: seed used (YOLOv5 hyperparameter)

Each training process corresponds to a specific time stamp `time_stamp’`, and outputs generated are
- subdirectory `analysis/output/time_stamp containing following:
  - `baseline_metrics.json`: baseline metrics
  - `classCM_losses_trn.json` and `classCM_losses_val.json`: classification `confusion matrices for training nad validation data
  - `configurations.txt`: training configurations
  - `data_info.txt`: data information
  - `dic_FitConfEp.json and dic_max_FitConfEp.json`: dictionary with fitness, confidence threshold maximising detection F1-score and epoch number for all epochs and for epoch yielding maximum fitness
  - `job_id.txt`: batch job ID number
  - `opt.yaml`: opt file containing parameters used in yolov5 framework
  - `plots`: subdirectory containing metric curve plots in `.png` format
- subdirectory `runs/train/time_stamp` with following
  - `configs`: contains training configurations (`configurations.txt`) and and hyperparamters (`hyp.yaml`) files
  - `data`: contains multiple `.txt` and `.yaml` (listing training and evaluation data) for usage `yolov5` framework
  - `job_id.txt`: contains batch job ID number
  - `results`: results output by yolov5 framework

For model evaluation, run script `evaluate_job.sh`, which in turn calls `evaluate.py`. Configurations are specified as a dictionary and are as follows:
- `'parent_dir'`: parent directory (e.g. `'/cluster/project/eawag/p05001/repos/greyHeronRecognition/'’`)
- `'imgsz'`: image resolution
- `'ls_cams'`: list of filtered cameras
- `'batch_size'`: batch size
- `'time_stamp'`: time stamp of training job originally or `'megadetector'`
- `'n_last_im'`: number of last images for background removal; if 'none', then original images are taken
- `'day_night'`: day or night-time images, or both
- `'resample'`: resample method applied to dataset
- `'split'`: first (`'chronological'`) or second (`'seasonal'`) split used
- `'workers'`: number of workers used for data loading
- `'best_last'`: load best ('best') or last ('last') model during training
- `'n_gpus'`: number of gpus used
- `'conf_tsh'`: specifies confidence threshold; if set to 
- `'get_config': extract confidence threshold saved for training configs
- `'get_stats': extract confidence threshold yielding maximum F1-score during training
- `'best': determine new confidence threshold yielding maximum F1-score during evaluation for specified model and dataset
- `‘n_cams_regroup'`: number of regrouped cameras for log oversampling 2
- `'which_set'`: training (`'trn'`), validation (`'val'`), both (`'trn_val'`) or test (`'tst'`)

Every evaluation procedure is associated with a time stamp (different to the one for training) and resulting outputs are found in `analysis/output_eval/time_stamp` as follows
- `configurations_eval.txt`: contains evaluation configurations
- `data`: subdirectory with `.txt`, .cache and.yaml files  (listing data) for usage `yolov5` framework
- `data_info.txt`: contains data information
- `job_id.txt`: contains batch job ID number
- `results`: results output by `yolov5` framework

Furthermore subdirectory `greyHeronDetection/detection` contains scripts a script that applies a trained or untrained object detection model to a specified dataset. More details are found in the corresponding `README.md` file
