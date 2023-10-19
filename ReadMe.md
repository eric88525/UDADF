# Breaking Boundaries in Retrieval Systems: Unsupervised Domain Adaptation with Denoise-Finetuning

This repository contains the implementation of a method for Breaking Boundaries in Retrieval Systems: Unsupervised Domain Adaptation with Denoise-Finetuning.

![](./imgs/flow.png)

## Introduction
We introduce a novel technique for unsupervised domain adaptation in information retrieval
(IR). Our approach focuses on adapting both the dense retrieval model and the rerank model within a comprehensive two-stage retrieval system.

Our adaptation pipeline consists of three parts:
1. Construction of pseudo training set.
2. Denoise-finetuning the rerank model.
3. Knowledge distillation from the rerank model to the dense retrieval model

## Folder structure

| Folder Name   | Description  |
|---------------|--------------|
| bm25          | Stores testset bm25 top1000 results           |
| config        | Configuration for creating pseudo dataset            |
| datasets      | Holds original datasets downloaded via the beir package            |
| denoise_ir    | All functionalities           |
| example       | Pseudo queries and training data for the paper           |
| experiment    | Experimental scripts related to the analysis section            |
| imgs          | Images in ReadMe |
| output        | Default directory for storing models            |
| preprocess    | Data preprocessing, downloading, or generation-related            |
| pseudo        | Default generation directory for storing generated questions and datasets            |
| scripts       | Main experiment-related scripts            |
| train_model   | Train cross-encoder and bi-encoder models          |


## Installation
Our experiment was conducted using a single RTX 3090.
```
conda create -n review python=3.8
conda activate review
pip install -r requirements.txt
```
## Pipeline

### Construction of pseudo training set.
+ Select a subset of passages from the corpus, input them into the Query generator to generate pseudo queries, forming (query, passage) pairs.
+ Apply cross-encoder to label (query, passage) pairs, creating (query, passage, score) triplets.
+ The `dataset` can be changed to (scifact, fiqa, trec-covid).
```
python preprocess/build_pseudo_dataset.py \
    --config config/${dataset}.yaml
```

### Denoise-finetuning the rerank model.

**Build $D_{ce}$ (in Section 3.3)**
+ From the scored (query, passage, score) triplets obtained in step one, select the top n (default=100k) highest-scoring (query, passage, score) triplets as positive samples for denoise fine-tuning.
+ Retrieve the top 1000 passages related to the queries using the BM25 and Dense retrieval models to create two negative pools, each with a size of 1000 passages.
+ Randomly select one passage each as negative samples from the negatives pool retrieved by both BM25 and the dense retrieval model.
+ Combine positive samples and negative samples to form a pseudo dataset.

![](./imgs/build_ps_dataset.png)
+ After create pseudo dataset $D_{ce} = \{(Q_i, P_i, y_i)\}, y \in \{0, 1\} $, we pseudo dataset the for fine-tuning the cross-encoder.
![](./imgs/denoise_finetune.png)
```, 
sh scripts/cross_encoder_adaptation.sh
```

### Knowledge Distillation from the Rerank Model to the Dense Retrieval Model

In this phase, knowledge is distilled from both the adapted cross-encoder and the unadapted cross-encoder models to the bi-encoder.

![](./imgs/kd.png)

**Labeling Margins**
+ Labeling margins are calculated using the formula: margin = (query, positive passage) - (query, negative passage).
+ Fill in the path of the adapted_cross_encoder into the cross-encoder trained in step two.

```bash
sh scripts/build_distillation_dataset.sh
```
| Parameter | Description |
| - | - |
| unadapted_cross_encoder | The cross-encoder model that has not undergone domain adaptation. |
| adapted_cross_encoder | The cross-encoder model that has been adapted to the domain. Please provide the cross-encoder model trained in the previous step. |
| path_to_generated_data | The folder containing pseudo queries. |



**Training bi-encoder with margins**
```
sh scripts/bi_encoder_adaptation.sh
```

## Experiment Reproduction

Within the "experiment" folder, you'll find the corresponding experiments. Just execute the following scripts:
+ Experiments 5-3, 5-4, 5-6, and 5-7 utilize the pseudo dataset provided in the "example" folder.
+ Experiment 5-2 will first run the process of creating a pseudo dataset. If you've already created the pseudo dataset, you can modify the `generate_pseudo_queries`, `score_pseudo_qrels`, and `create_cross_train_data` settings to false within the "config" folder.

```
experiment
├── 5-2_Random_Seed.sh
├── 5-3_Model_Size.sh
├── 5-4_Sequence_Length.sh
├── 5-6_Noisy_Dataset_Simulation.sh
└── 5-7_The_Impact_of_r.sh
```

**5-2 Random Seed**

Run the experiment three times with different random seeds.

**5-3 Model Size**

Compare the differences between the L6, L12 model in denoise fine-tuning and normal fine-tuning.

**5-4 Sequence Length**

Examine the impact of limiting input sequence length.

**5-6 Noisy Dataset Simulation**

Invert {1%, 5%, 10%} of the pseudo dataset and compare denoise fine-tuning with normal fine-tuning.

**5-7 The Impact of r**

Set r = {0, 1, 5, 10, 20} and compare by experimenting with the inversion of 20% of the pseudo dataset.
