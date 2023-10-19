import torch
import numpy as np
import random
import os
import json
from beir.datasets.data_loader import GenericDataLoader
import logging

# Set the random seed for reproducibility
def set_seed(seed):
    """
    Set the random seed for reproducibility.

    Args:
        seed (int): The seed value to set for random number generators.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# Join prefix and name with a hyphen separator
def prefix_join(prefix, name):
    """
    Join prefix and name with a hyphen separator.

    Args:
        prefix (str): The prefix string.
        name (str): The name string.

    Returns:
        str: The joined string.
    """
    if prefix:
        return prefix + "-" + name
    else:
        return name

# Load the corpus of documents
def load_corpus(data_folder: str, remove_empty=True):
    """
    Load the corpus of documents from the specified data folder.

    Args:
        data_folder (str): The path to the data folder.
        remove_empty (bool, optional): Whether to remove empty documents from the corpus. Defaults to True.

    Returns:
        dict: The loaded corpus of documents.
    """
    corpus = GenericDataLoader(data_folder).load_corpus()

    logging.info("Loaded corpus with {} documents".format(len(corpus)))
    print("Loaded corpus with {} documents".format(len(corpus)))
    
    if remove_empty:
        corpus = {k: v for k, v in corpus.items() if v["text"] != ""}
        logging.info("Removed {} empty documents".format(len(corpus)))
        print("Removed {} empty documents".format(len(corpus)))
    
    if remove_empty:
        for doc_id, doc in corpus.items():
            assert doc["text"] != "", f"Empty document found: {doc_id}"

    return corpus

# Load the queries from a JSONL file
def load_queries(data_folder, prefix="gen"):
    """
    Load the queries from a JSONL file.

    Args:
        data_folder (str): The path to the data folder.
        prefix (str, optional): The prefix for the query file. Defaults to "gen".

    Returns:
        dict: The loaded queries.
    """
    query_file = os.path.join(data_folder, prefix_join(prefix, "queries.jsonl"))
    assert os.path.exists(query_file), f"{query_file} not found"
    
    queries = {}
    with open(query_file, encoding="utf8") as fIn:
        for line in fIn:
            line = json.loads(line)
            queries[line.get("_id")] = line.get("text")
    
    return queries
