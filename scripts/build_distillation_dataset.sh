#!/bin/bash

unadapted_cross_encoder="cross-encoder/ms-marco-MiniLM-L-12-v2"

# please fill the path to the adapted cross-encoder
adapted_cross_encoder=""
path_to_generated_data="pseudo/scifact"

python preprocess/build_distillation_dataset.py \
    --unadapted_cross_encoder $unadapted_cross_encoder \
    --adapted_cross_encoder $adapted_cross_encoder \
    --path_to_generated_data $path_to_generated_data