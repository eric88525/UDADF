#!/bin/bash

dataset="scifact"

# the path to the distill dataset
distill_dataset_path=pseudo/${dataset}/distill_dataset.tsv

path_to_generated_data="pseudo/$dataset"
original_data_folder="datasets/$dataset"

python train_model/train_dual_by_distill.py \
    --dataset=$dataset \
    --path_to_generated_data=$path_to_generated_data \
    --original_data_folder=$original_data_folder \
    --distill_dataset_path=$distill_dataset_path