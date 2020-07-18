# Architopes: An Architecture Modification for Composite Pattern Learning, Increased Expressiveness, and Reduced Training Time

## Requirements

To install requirements:
*  Install [Anaconda](https://www.anaconda.com/products/individual)  version 4.8.2.
* Create Conda Environment
``` pyhton
# cd into the same directory as this README file.

conda create python=3.7.6 --name architopes \
conda activate architopes \
pip install -r requirements.txt
```


## Training and Evaluation
1. Preprocess

* In this step, latitude and longitude are mapped to Euclidean coordinates about the projected extrinsic mean to remove the effect of earth’s curvature and then the data is split into distinct set using the proximity to ocean. After each set is split into train and test using 30% for test.
Skip this step to use the same exact train and test sets used to produce teh results in the paper.
``` python
python3 ./code/preprocessing.py \
--source_file './data/raw/housing.csv' \
--sink_path './data/data'
```
2. Specify the parameters related to each set and the space of  hyper parameters in ./Training_Evaluation/Grid.py. Default parameters used to obtain the results in the paper can be found in the appendix.

3. Train and save the results for the ffNN and ffNN-tope using the following command:
``` python
python3 ./code/train_eval.py \
--is_test 'F'  \
--is_manual 'T' \
--n_iter 200 \
--n_jobs 30 \
--result_path './results/train_eval' \
--data_path './data/data' 
```
4. Compile Results
```sh
python ./code/postprocessing.py \
--result_path './results/train_eval' \
--compile_path './results/compiled' \
```
## Pre-trained Models

Pre-trained models are found in `./pretrained_models`. `pretrained_models.ipynb` is provided for easier use.


## Results

Our model achieves the following performance on :

### [California Housing Price Dataset](https://github.com/ageron/handson-ml/tree/master/datasets/housing)

The house prices were multiplied by 10^(-5) to avoid exploding gradient issues.

1. For Train:

| Model name         | MAE             | MAPE           | MSE |
| ------------------ |---------------- | -------------- |-----|
| ffNN-tope          |  0.285          |15.01           |0.210|
| ffNN               |  0.297          |16.41           |0.211|
| ffNN-dp            |  0.411          |19.94           |0.395| 

1. For Test:

| Model name         | MAE             | MAPE           | MSE |
| ------------------ |---------------- | -------------- |-----|
| ffNN-tope          | 0.306           | 16.33          |0.232|
| ffNN               |  0.322          | 18.36          |0.244|
| ffNN-dp            |  0.413          | 20.16          |0.398|




