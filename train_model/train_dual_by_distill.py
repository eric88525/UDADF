import argparse
import logging
import os
import glob
import torch
from torch.utils.data import DataLoader
from datetime import datetime
import json
import sys

def train(args):
    
    with open(os.path.join(args.output_dir, "params.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    logger.info(f"Save model at: {args.output_dir}")

    # Load the generated data
    corpus = load_corpus(args.original_data_folder, remove_empty=False)
    gen_queries = load_queries(args.path_to_generated_data, prefix=args.prefix)

    logger.info(
        "Now doing training on the generated data with the MarginMSE loss")

    # It can load checkpoints in both SBERT-format (recommended) and Huggingface-format
    model: SentenceTransformer = load_sbert(
        args.base_ckpt, args.pooling, args.max_seq_length)

    
    assert os.path.isfile(args.distill_dataset_path)
    logger.info(f"Load distillation training data from {args.distill_dataset_path}")

    train_dataset = GenerativePseudoLabelingDataset(
        args.distill_dataset_path, gen_queries, corpus)
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=False,
        batch_size=args.batch_size_gpl,
        drop_last=True,
    )  # Here shuffle=False, since (or assuming) we have done it in the pseudo labeling

    logger.info(f"Dataset size: {len(train_dataset)}")

    train_loss = MarginDistillationLoss(
        model=model, similarity_fct=args.gpl_score_function)

    model._target_device = torch.device(args.device)
    # assert args.gpl_steps > 1000
    model.fit(
        [(train_dataloader, train_loss)],
        epochs=args.epoch,
        steps_per_epoch=args.gpl_steps,
        warmup_steps=args.warmup_steps,
        checkpoint_save_steps=10000,
        checkpoint_save_total_limit=10000,
        output_path=args.output_dir,
        checkpoint_path=args.output_dir,
        use_amp=args.use_amp,
        optimizer_params={
            "lr": args.lr,
        }
    )

def eval(args):
    from gpl.toolkit import evaluate
    logger.info("Doing evaluation for adapted bi-encoders")

    for ckpt_dir in glob.glob(f"{args.output_dir}/*"):
        print(f"evaluating {ckpt_dir}")
        try:
            assert os.path.isdir(ckpt_dir) and "pytorch_model.bin" in os.listdir(
                ckpt_dir)  # make sure it's a checkpoint dir
            evaluate(
                data_path=args.original_data_folder,
                output_dir=ckpt_dir,
                model_name_or_path=ckpt_dir,
                max_seq_length=args.max_seq_length,
                score_function=args.gpl_score_function,
                pooling=args.pooling,
                k_values=[10, 50, 100, 1000],
                split="test",
            )
        except Exception as e:
            print(e)

def get_args():
    parser = argparse.ArgumentParser(description="Your Script Description")

    parser.add_argument("--base_path", default=os.getcwd(), help="Base path")
    parser.add_argument("--dataset", help="Dataset name", choices=["trec-covid", "fiqa", "scifact"])
    parser.add_argument("--prefix", default="gen", help="Prefix")
    parser.add_argument("--output_dir", default=None, help="Model output directory")
    parser.add_argument("--gpl_score_function", default="dot", help="GPL score function")
    parser.add_argument("--path_to_generated_data", help="Path to generated data")
    parser.add_argument("--original_data_folder", help="Original data folder")
    parser.add_argument("--distill_dataset_path", help="Distill dataset path")
    parser.add_argument("--base_ckpt", default="distilbert-base-uncased", help="Base checkpoint")
    parser.add_argument("--pooling", default="mean", help="Pooling method")
    parser.add_argument("--max_seq_length", type=int, default=350, help="Max sequence length")
    parser.add_argument("--batch_size_gpl", type=int, default=32, help="Batch size for GPL")
    parser.add_argument("--gpl_steps", type=int, default=140000, help="Number of GPL steps")
    parser.add_argument("--use_amp", type=bool, default=True, help="Use automatic mixed precision")
    parser.add_argument("--device", default="cuda:0", help="Device for training")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--epoch", type=int, default=1, help="Number of epochs")
    parser.add_argument("--warmup_steps", type=int, default=10000, help="Warmup steps")

    return parser.parse_args()

if __name__ == "__main__":

    args = get_args()
    
    if args.output_dir is None:
        args.output_dir = os.path.join(
            args.base_path, "output", "bi_encoder", args.dataset, datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        )
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Just some code to print debug information to stdout
    logging.basicConfig(filename=os.path.join(args.output_dir, "train.log"),
                        format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        )
    
    logger = logging.getLogger(__name__)
    
    sys.path.append(os.getcwd())
    from denoise_ir.utils import *
    from gpl.toolkit import (
        GenerativePseudoLabelingDataset,
        MarginDistillationLoss,
        load_sbert,
    )
    from sentence_transformers import SentenceTransformer
    
    train(args)
    eval(args)