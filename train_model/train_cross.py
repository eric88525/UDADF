'''
https://github.com/UKPLab/sentence-transformers/blob/master/examples/training/ms_marco/train_cross-encoder_scratch.py
'''
# %%
from beir.reranking.models import CrossEncoder
import transformers
import numpy as np
import torch
from typing import Any, List, Dict, Tuple
from torch.utils.data import DataLoader
from datetime import datetime
import argparse
import random
import logging
import json
import gzip
import os
from torch.utils.data import Dataset
from sentence_transformers.cross_encoder.evaluation import CERerankingEvaluator
import sentence_transformers.util as sent_util
from beir import util, LoggingHandler
from sentence_transformers import InputExample
import sys
import os
sys.path.append(os.getcwd())

transformers.logging.set_verbosity_error()

def main(args):
    from denoise_ir.model import DenoiseCrossEncoder
    from denoise_ir.dataset import PseudoCrossDataset
    from denoise_ir.evaluator import TestEvaluator

    # save args
    with open(os.path.join(model_save_path, "args.txt"), "w") as fOut:
        json.dump(args.__dict__, fOut, indent=2)

    train_dataset = PseudoCrossDataset(
        generated_path=args.generated_path,
        prefix=args.prefix,
        hard_negatives_name=args.hard_negatives_name,
        skip_top_n=args.skip_top_n,
        use_top_n=args.use_top_n,
        neg_retrievers=args.neg_retrievers,
    )

    # use existing train pairs
    if args.train_pairs_path is not None and os.path.exists(args.train_pairs_path):
        train_dataset.load_train_pairs(
            args.train_pairs_path, unique=args.unique)
    else:
        train_dataset.create_topn_train_samples(**vars(args))
        logging.info(f"Saving train pairs to {args.train_pairs_path}")

        # save train pairs to target path
        if args.train_pairs_path is not None:
            PseudoCrossDataset.save_train_pairs(
                fpath=args.train_pairs_path,
                train_pairs=train_dataset.train_pairs
            )
    # save train pairs to model save path
    PseudoCrossDataset.save_train_pairs(
        fpath=os.path.join(
            model_save_path, "train_pairs.json"),
        train_pairs=train_dataset.train_pairs
    )

    # Test model during training
    test_eval = TestEvaluator(
        retrieval_result=args.test_retrieval_result,
        topk=[100],
        dataset=args.test_dataset,
        k_values=[10, 50, 100],
        test_matrix="NDCG@10",
        batch_size=512,
        max_test_samples=args.max_test_samples,
    )

    # add any model here
    model_name_mapping = {
        "L12": "cross-encoder/ms-marco-MiniLM-L-12-v2",
        "L6": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    }
    # if you want to use multiple models for denoise fine-tuning, use @ to separate them
    # e.g. L12@L12
    model_names = [model_name_mapping[m] for m in args.model_name.split("@")]
    logging.info(f"Model names: {model_names}")

    model = DenoiseCrossEncoder(model_names=model_names,
                                num_labels=1,
                                max_length=args.max_seq_length,
                                device=args.device,
                                test_evaluator=test_eval)

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.train_batch_size)

    warmup_steps = int(args.num_epochs *
                       len(train_dataloader) * args.warmup_ratio)
    logging.info("Warmup-steps: {}".format(warmup_steps))

    global_steps = args.num_epochs * len(train_dataloader)
    if args.denoise_warmup_ratio is not None and len(model_names) > 1:
        denoise_warmup_steps = int(global_steps * args.denoise_warmup_ratio)
    else:
        denoise_warmup_steps = 0
    logging.info("Denoise warmup-steps: {}".format(denoise_warmup_steps))

    if args.random_batch_warmup_ratio > 0:
        random_batch_warmup_steps = int(
            args.random_batch_warmup_ratio * global_steps)
    else:
        random_batch_warmup_steps = 0
    logging.info("Random batch warmup-steps: {}".format(random_batch_warmup_steps))

    model.fit(train_dataloader=train_dataloader,
              gamma=args.gamma,
              denoise_warmup_steps=denoise_warmup_steps,
              random_batch_warmup_steps=random_batch_warmup_steps,
              random_batch_warmup_p = args.random_batch_warmup_p,
              epochs=args.num_epochs,
              evaluation_steps=args.evaluation_steps,
              warmup_steps=warmup_steps,
              output_path=model_save_path,
              use_amp=True,
              optimizer_params={'lr': args.lr},
              show_progress_bar=False,
              )

def get_args():

    parser = argparse.ArgumentParser()


    parser.add_argument("--exp_flag", type=str, default="cross_encoder")
    parser.add_argument("--seed", type=int, default=2023)
    ##################################################
    parser.add_argument("--denoise_warmup_ratio", type=float, default=0.1)
    parser.add_argument("--random_batch_warmup_ratio", type=float, default=0.1)
    parser.add_argument("--random_batch_warmup_p", type=float, default=0.2)
    parser.add_argument("--gamma", type=float, default=1.0)
    ##################################################
    # set hard negatives and qrel name
    parser.add_argument("--hard_negatives_name",
                        type=str, default="hard-negatives.jsonl")
    parser.add_argument("--prefix", type=str, default="gen")
    parser.add_argument("--train_pairs_path", type=str, default=None,
                        help="Path to the train pairs")
    parser.add_argument(
        "--unique", type=lambda x: (str(x).lower() == 'true'), default=False)
    ################## negative sampling #############
    parser.add_argument("--neg_per_model", type=int, default=0,
                        help="Number of negative samples per model")
    parser.add_argument("--skip_top_n", type=int, default=10,
                        help="Number of top-n samples to skip")
    parser.add_argument("--use_top_n", type=int, default=50,
                        help="Number of top-n samples to use")
    parser.add_argument("--sample_mode", type=str, default="bottom",
                        choices=["random", "bottom"], help="Sampling mode")
    parser.add_argument("--neg_retrievers", nargs='+', default=None)
    ##################################################
    parser.add_argument("--base_path", type=str, default=os.getcwd(),
                        help="Base path for the project")
    parser.add_argument("--generated_path", type=str,
                        help="Path to the generated data")
    parser.add_argument("--dataset", type=str,
                        default="trec-covid", help="Dataset name")
    parser.add_argument("--device", type=str,
                        default="cuda:0", help="Device to use")
    parser.add_argument("--pos_neg_ration", type=int,
                        default=2, help="Number of negative samples per positive sample")
    ##################################################
    parser.add_argument("--lr", type=float,
                        default=1e-5, help="Learning rate")
    ##################################################
    parser.add_argument("--model_name", type=str,
                        default="L12", help="Model name")
    parser.add_argument("--max_seq_length", type=int,
                        default=300, help="Max sequence length")
    parser.add_argument("--train_batch_size", type=int,
                        default=32, help="Train batch size")
    parser.add_argument("--num_epochs", type=int,
                        default=2, help="Number of epochs")
    parser.add_argument("--warmup_ratio", type=float,
                        default=0.1, help="Warmup ratio")
    parser.add_argument("--evaluation_steps", type=int,
                        default=1000, help="Evaluation steps")
    parser.add_argument("--use_amp", type=bool, default=True, help="Use AMP")
    ##################################################
    parser.add_argument("--test_retrieval_result", type=str, default=None)
    parser.add_argument("--test_dataset", type=str, default=None)
    parser.add_argument("--max_test_samples", type=int, default=100)
    args = parser.parse_args()

    return args


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

if __name__ == "__main__":

    args = get_args()
    print(args)

    # Provide model save path
    model_save_path = os.path.join(
        args.base_path, "output", args.exp_flag, args.dataset, f"{args.model_name.replace('/','-')}-{datetime.now().strftime('%m-%d_%H-%M-%S')}")

    print("Model save path: ", model_save_path)
    
    os.makedirs(model_save_path, exist_ok=True)

    # Just some code to print debug information to stdout
    logging.basicConfig(filename=os.path.join(model_save_path, "train_log.txt"),
                        format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        )

    set_seed(int(args.seed))
    main(args)