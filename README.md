Code for the paper *Data-Driven Approach to Mitigate Acoustic Sensor Positioning Effects in LPBF*

## Subfolder
- CIRP CMMO 2025
```
git clone https://git@git.overleaf.com/66eb1c514e9aebd36d6d5466 "CIRP CMMO 2025"
```
- lfs
<!-- ``` -->
<!-- git clone https://huggingface.co/mingwucn/Data-Driven_Approach_to_Mitigate_Acoustic_Sensor_Positioning_Effects_in_LPBF "lfs" -->
<!-- ``` -->
baidu pan:
DATA/lfs/"Data-Driven Approach to Mitigate Acoustic Sensor Positioning Effects in LPBF"


## Preprocessing
1. `read_data.ipynb`: A brief overview of the DAQ/LMQ structures
2. `preprocessing_0_cube_slicer.ipynb`: Segment the cube, as all the cubes were printed in one go. 

## Statistic difference due to the position and direction
`statistical_difference.ipynb`
`statistical_difference.py`

## Prepare for training a classifier to classify the context
Input: FT of time series
Output: Direction (Converted to one hot encoder)

## Postprocessing
1. `accuracy`: The overall accuracy of trained results
2. `grad-cam`:

## Training script
`python train_dist_1d.py --gpu=0 --save_every=10 --fold_i=0 --folds=10 --model_name=CNN --batch_size=1024 --learning_rate=5e-4 --repeat=1 --epochs=50 --num_workers=5 --input_type='mic+energy' --output_type='direction'`

## CNN
Regime classification using line-wise labels
1. Construct line-wise labels via `construct_labels.py` generate a `lpbf_line_wise_data` in the `$DAQDir/intermediate`
2. Check the model with `pre_train.ipynb`