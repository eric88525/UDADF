# %%
from beir.datasets.data_loader import GenericDataLoader
import os
import sys
from beir import util, LoggingHandler
import logging
import argparse

def main(args):

    from denoise_ir.faster_pl import PseudoLabeler
    from denoise_ir.mine import NegativeMiner

    miner = NegativeMiner(
        args.path_to_generated_data,
        prefix=args.prefix,
        batch_size=512
    )
    miner.run(output_path=os.path.join(
        args.path_to_generated_data, "dual-hard-negatives.jsonl"))

    corpus, gen_queries, _ = GenericDataLoader(
        args.path_to_generated_data,
        prefix=args.prefix,

    ).load(split="train")

    cross_encoders = [args.unadapted_cross_encoder, args.adapted_cross_encoder]
    # %%
    pseudo_labeler = PseudoLabeler(
        args.path_to_generated_data,
        gen_queries,
        corpus,
        total_steps=140000,
        batch_size=32,
        cross_encoders=cross_encoders,
        max_seq_length=300,
        hard_negatives='dual-hard-negatives.jsonl',
        cross_encoder_batch_size=1300,
        device="cuda:0",
    )

    pseudo_labeler.output_path = os.path.join(
        args.path_to_generated_data, "distill_dataset.tsv")
    pseudo_labeler.run()

def get_args():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--unadapted_cross_encoder", type=str, default="cross-encoder/ms-marco-MiniLM-L-12-v2")
    parser.add_argument("--adapted_cross_encoder", type=str, help="path to adapted cross encoder")
    parser.add_argument("--path_to_generated_data", type=str)
    parser.add_argument("--prefix", type=str, default="gen")
    args = parser.parse_args()
    
    return args

if __name__ == "__main__":

    sys.path.append(os.getcwd())
    args = get_args()
    print(args)
    
    logging.basicConfig(filename=os.path.join(args.path_to_generated_data, "distill_log.txt"),
                        format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        )
    main(args)