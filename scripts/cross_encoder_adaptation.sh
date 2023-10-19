#!/bin/bash
# Script to fine-tune a model for denoising and normal tasks.

# Set the base path to the current working directory.
base_path="$(pwd)"
seed=1

# Denoising Fine-Tuning Loss (Eq. 3)
# Loss = L_label + gamma * L_agg
# T = total training steps
# After T * denoise_warmup_ratio training steps, the model starts co-regularization learning.
denoise_warmup_ratio=0.1

# During the first T * {random_batch_warmup_ratio} training steps, 
# the model randomly drops batches with a probability of {random_batch_warmup_p}.
random_batch_warmup_ratio=0.1 
random_batch_warmup_p=0.2 # Probability of batch dropout

# Learning rate warm-up
warmup_ratio=0.1 
gamma=1 # r in Eq. 3

# Hard negatives and qrel name
hard_negatives_name="cross-hard-negatives.jsonl"
prefix="gen" # prefix of the generate queries
unique=false # If true, use only unique (query, passage, label) triples

# Positive/Negative Sampling
neg_per_model=1 # Number of negative passages from each model
pos_neg_ratio=2 # Number of negative passages / Number of positive passages
neg_retrievers="bm25 msmarco-distilbert-base-v3" # Retrievers used to sample negative passages, separated by space

# Using passages in the range (skip_top_n, use_top_n) as hard negatives
skip_top_n=0
use_top_n=1000
sample_mode="random"

# CUDA device
device="cuda:0"

# Training arguments
lr=1.5e-5 # Learning rate
train_batch_size=32 # Batch size
num_epochs=2 # Number of epochs

# Model arguments
# You can add more models at train_model/train_cross.py. 
# Add {model_name:path} to the model_name_mapping dictionary.
# To use two models for co-regularization, separate them with @.
# For example, model_name="L12@L12" or model_name="L6@L12"
model_name="L12" 
max_seq_length=300 # Maximum length of input sequence
use_amp=True # Use automatic mixed precision training

# Evaluation arguments
dataset="scifact"
test_dataset="scifact"
test_retrieval_result="bm25/${dataset}_test_top1000.json"
max_test_samples=10000 # Number of samples used for model evaluation
evaluation_steps=1000 # Evaluate the model every {evaluation_steps} training steps
generated_path="pseudo/${dataset}" # Path for pseudo dataset

# Run normal fine-tuning and denoise fine-tuning
for finetune_method in "normal" "denoise"; do

    if [ "${finetune_method}" = "normal" ]; then
        # Normal fine-tuning
        model_name="L12"
        denoise_warmup_ratio=0
        random_batch_warmup_ratio=0
        warmup_ratio=0.1 # Learning rate warm-up
        gamma=0
    else
        # Denoise fine-tuning
        model_name="L12@L12"
        denoise_warmup_ratio=0.1
        random_batch_warmup_ratio=0.1 # After T * random_batch_warmup_ratio, the model starts co-regularization learning
        warmup_ratio=0.1 # Learning rate warm-up
        gamma=1
    fi

    python train_model/train_cross.py \
    --seed $seed \
    --denoise_warmup_ratio $denoise_warmup_ratio \
    --random_batch_warmup_ratio $random_batch_warmup_ratio \
    --random_batch_warmup_p $random_batch_warmup_p \
    --gamma $gamma \
    --hard_negatives_name $hard_negatives_name \
    --prefix $prefix \
    --unique $unique \
    --neg_per_model $neg_per_model \
    --skip_top_n $skip_top_n \
    --use_top_n $use_top_n \
    --sample_mode $sample_mode \
    --base_path $base_path \
    --generated_path $generated_path \
    --dataset $dataset \
    --test_dataset $test_dataset \
    --device $device \
    --pos_neg_ratio $pos_neg_ratio \
    --lr $lr \
    --model_name $model_name \
    --max_seq_length $max_seq_length \
    --train_batch_size $train_batch_size \
    --num_epochs $num_epochs \
    --warmup_ratio $warmup_ratio \
    --evaluation_steps $evaluation_steps \
    --use_amp $use_amp \
    --neg_retrievers $neg_retrievers \
    --dataset $dataset \
    --test_retrieval_result $test_retrieval_result \
    --max_test_samples $max_test_samples
done
