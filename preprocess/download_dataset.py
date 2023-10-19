import argparse
import os
import sys

if __name__ == "__main__":

    sys.path.append(os.getcwd())
    from denoise_ir.preprocess import download_dataset

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["trec-covid", "scifact", "fiqa"])
    parser.add_argument("--base_path", type=str, default=os.getcwd())
    args = parser.parse_args() 
    
    download_dataset(args.dataset, args.base_path)