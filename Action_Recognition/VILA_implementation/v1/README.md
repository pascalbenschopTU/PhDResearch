# VILA v1

## Installation
Clone VILA
Setup dependencies for VILA
Copy files from this folder to VILA

Environment in environment.yml is used for predictions

## Running own server

```bash
cd Action_Recognition/VILA/
conda activate vila

python -W ignore server_v2.py
```

## Running predictions

```bash

python predictions.py -d {dataset_name} -m {model_name} -e {eval_json} -c {class_to_test}

dataset_name and model_name is used for the dir that the eval json is in
eval_json is optional and can be used to load a json file with predictions
class_to_test is optional and limits the videos to the class of interest
```