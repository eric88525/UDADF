
import logging
import os
from beir import LoggingHandler
import yaml
import argparse
import sys

logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[LoggingHandler()],
)

def main(config):
    
    from denoise_ir.preprocess import download_dataset, resize, generate_pseudo_queries, score_pseudo_qrels, create_cross_train_data, create_dev
    from denoise_ir.utils import set_seed
    
    set_seed(config["seed"])

    # download dataset from BEIR benchmark, save at datasets/{dataset}
    data_folder = download_dataset(config["dataset"], config["base_path"])

    if config["pseudo_data_folder"] is not None:
        pseudo_data_folder = config["pseudo_data_folder"]
    else:
        pseudo_data_folder = os.path.join(config["base_path"], "pseudo",
                                          config["dataset"])

    if os.path.exists(pseudo_data_folder) and  \
            'corpus.jsonl' in os.listdir(pseudo_data_folder):
        logging.info("Pseudo data folder already exists, skipping resize.")
    else:
        resize(data_folder, pseudo_data_folder, config["new_corpus_size"])

    logging.info("Data folder: {}".format(data_folder))
    logging.info("Pseudo data folder: {}".format(pseudo_data_folder))

    # generate_pseudo_queries, save at {pseudo_data_folder}
    if config["generate_pseudo_queries"]:
        generate_pseudo_queries(
            pseudo_data_folder,
            prefix=config["prefix"],
            batch_size=config["query_gen"]["batch_size"],
            model_path=config["query_gen"]["model_path"],
            device=config["device"],
        )

    # use {cross_encoder_name} cross-encoder to label the relevance of (query, passage) pairs
    if config["score_pseudo_qrels"]:
        score_pseudo_qrels(
            pseudo_data_folder,
            prefix=config["prefix"],
            cross_config={
                "model_name": config["cross_train_data"]["cross_encoder_name"],
                "device": config["device"],
                "batch_size": config["cross_train_data"]["cross_batch_size"],
                "max_length": config["cross_train_data"]["max_length"],
            }
        )

    # create train data for fine-tune cross-encoder,
    # only use {top_n} highest scored (query, passage) pairs
    if config["create_cross_train_data"]:
        create_cross_train_data(
            pseudo_data_folder,
            original_data_folder=config["original_data_folder"],
            prefix=config["prefix"],
            top_n=config["cross_train_data"]["top_n"],
            limit_score=config["cross_train_data"]["limit_score"],
            nneg=config["cross_train_data"]["nneg"],
            batch_size=config["cross_train_data"]["mine_batch_size"],
            device=config["device"],
            hard_neg_name=config["cross_train_data"]["hard_neg_name"],
            retrievers=["msmarco-distilbert-base-v3", "bm25"],
        )


if __name__ == "__main__":


    sys.path.append(os.getcwd())

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    args = parser.parse_args()

    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    main(config)
