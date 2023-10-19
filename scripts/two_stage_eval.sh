#!/bin/bash

# models
rerank_model=""  # fill the path of the rerank model
retrieval_model="" # fill the path of the retrieval model

# dataset to eval
dataset="scifact"
ori_dataset="datasets/${dataset}"
split="test"


# batch size
retrieval_batch_size=512
rerank_batch_size=1300


python evaluation/two_stage.py \
    --rerank_model $rerank_model \
    --retrieval_model $retrieval_model \
    --dataset $dataset \
    --ori_dataset $ori_dataset \
    --split $split \
    --retrieval_batch_size $retrieval_batch_size \
    --rerank_batch_size $rerank_batch_size