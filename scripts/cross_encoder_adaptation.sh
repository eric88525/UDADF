#!/bin/bash
base_path="$(pwd)"
seed=1

# Denoising finetuning Loss = L_label + gamma * L_agg
# T = total training steps
denoise_warmup_ratio=0.1
random_batch_warmup_ratio=0.1 # after T * random_batch_warmup_ratio, model start to do co-regularization learning # a%
random_batch_warmup_p=0.2 # a% in in Figure 3.4
warmup_ratio=0.1 # learning rate warmup
gamma=1 # r

# Hard negatives and qrel name
hard_negatives_name="cross-hard-negatives.jsonl"
prefix="gen"
unique=false

# Pos/Negative sampling
neg_per_model=1
pos_neg_ratio=2
neg_retrievers="bm25 msmarco-distilbert-base-v3"

# using passage in (skip_top_n, use_top_n) as hard negatives
skip_top_n=0
use_top_n=1000
sample_mode="random"

# Paths
device="cuda:0"

# Training args
lr=1.5e-5
train_batch_size=32
num_epochs=2

# Model args
model_name="L12"
max_seq_length=300
use_amp=True

# evaluation args
dataset="scifact"
test_dataset="scifact"
test_retrieval_result="bm25/${dataset}_test_top1000.json"
max_test_samples=10000
evaluation_steps=1000
generated_path="pseudo/${dataset}"

for finetune_method in  "normal" "denoise"; do

    if [ "${finetune_method}" = "normal" ]; then
        # normal fine-tuning
        model_name="L12"
        denoise_warmup_ratio=0
        random_batch_warmup_ratio=0
        warmup_ratio=0.1 # learning rate warmup
        gamma=0
    else
        # denoise fine-tuning
        model_name="L12@L12"
        denoise_warmup_ratio=0.1
        random_batch_warmup_ratio=0.1 # after T*random_batch_warmup_ratio, model start to do co-regularization learning
        warmup_ratio=0.1 # learning rate warmup
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

