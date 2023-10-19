# %%
from typing import List
from torch.utils.data import Dataset
from beir.datasets.data_loader import GenericDataLoader
import os
import json
from collections import defaultdict
import random
from sentence_transformers import InputExample
import logging
from beir import LoggingHandler
from functools import reduce
from tqdm import tqdm

class PseudoCrossDataset(Dataset):
    def __init__(self, generated_path,
                 prefix="gen",
                 hard_negatives_name="hard-negatives.jsonl",
                 skip_top_n=0,
                 use_top_n=100,
                 neg_retrievers = ["bm25", "msmarco-distilbert-base-v3"]
                 ):

        self.corpus, self.gen_queries, self.gen_qrels = GenericDataLoader(
            generated_path, prefix=prefix).load(split="train")
        self.generated_path = generated_path
        self.skip_top_n = skip_top_n
        self.use_top_n = use_top_n
        self.train_pairs = []
        self.hard_negatives = None
        self.hard_negatives_name = hard_negatives_name
        self.neg_retrievers = neg_retrievers
        
    def get_doc_text(self, doc_id: str):
        doc_id = str(doc_id)
        return self.corpus[doc_id]["title"] + ' ' + self.corpus[doc_id]["text"]
    
    @staticmethod
    def load_hard_negatives(hard_negatives_path:str, neg_retrievers:List[str] = None):

        assert os.path.exists(
            hard_negatives_path), f"Hard negatives path {hard_negatives_path} does not exist"

        logging.info(f"Loading hard negatives, path: {hard_negatives_path} , neg_retrievers: {neg_retrievers}")
        hard_negatives = defaultdict(dict)

        with open(hard_negatives_path, "r") as f:
            for line in f:
                row = json.loads(line)
                """
                row = {
                    'qid': str,
                    'pos': list,
                    'neg': dict of {model_name: list}
                }
                """
                if neg_retrievers is not None:
                    row["neg"] = {k: v for k, v in row["neg"].items() if k in neg_retrievers}
                
                if row["qid"] not in hard_negatives:
                    hard_negatives[row["qid"]] = {
                        "pos": set(),
                        "model_negatives": row["neg"]
                    }
                hard_negatives[row["qid"]]["pos"].update(row["pos"])

        return hard_negatives

    @staticmethod
    def get_train_pair_name(seed, neg_per_query, neg_per_model, pos_neg_ration, skip_top_n, use_top_n, top_n_ensemble, **args):
        train_pair_name = f"s{seed}_{neg_per_query}_{neg_per_model}_{pos_neg_ration}_{skip_top_n}_{use_top_n}_{top_n_ensemble}_train_pairs.json"
        return train_pair_name

    @staticmethod
    def check_train_pairs(train_pairs: List[int]):

        logging.info(f"Checking train pairs")

        assert all([len(pair) == 3 for pair in train_pairs]
                   ), "Train pairs should be a list of [query, doc, label]"
        assert all([pair[2] in [0, 1] for pair in train_pairs]
                   ), "Train pairs label should be 0 or 1"

        q_doc_label = {}

        for q, doc, label in train_pairs:

            if f"{q} {doc}" not in q_doc_label:
                q_doc_label[f"{q} {doc}"] = label
            else:
                assert q_doc_label[f"{q} {doc}"] == label, f"{q} {doc} has different labels"

        logging.info(f"Checking train pairs done")

    @staticmethod
    def save_train_pairs(fpath, train_pairs):
        logging.info(f"Saving train pairs {fpath}")
        os.makedirs(os.path.dirname(fpath), exist_ok=True)

        with open(fpath, 'w') as f:
            json.dump(train_pairs, f)

    def load_train_pairs(self, fpath, unique=False):
        with open(fpath, 'r') as f:
            self.train_pairs = json.load(f)

        self.check_train_pairs(self.train_pairs)

        # remove duplicates pairs
        if unique:
            logging.info(f"Removing duplicate pairs")
            unique_list = list(set(map(tuple, self.train_pairs)))
            self.train_pairs = [list(item) for item in unique_list]

        # count pos and neg labels
        pos, neg = 0, 0
        for pair in self.train_pairs:  # pair = [query, doc, label]
            if pair[2] == 1:
                pos += 1
            else:
                neg += 1

        logging.info(f"Loading train pairs {fpath}")
        logging.info(f"Labeled {len(self.train_pairs)} pairs")
        logging.info(f"Pos: {pos}, Neg: {neg}")


    def create_topn_train_samples(self, neg_per_model, pos_neg_ration, sample_mode: str, pos_label=1, neg_label=0, **args):

        if self.hard_negatives is None:
            self.hard_negatives = self.load_hard_negatives(
                os.path.join(self.generated_path, self.hard_negatives_name), self.neg_retrievers)        

        assert sample_mode in ["random", "bottom"]

        logging.info(f"Use top-n: {self.use_top_n}")

        self.train_pairs = []
        pos_counts, neg_counts = 0, 0

        for qid in tqdm(self.hard_negatives):

            pos = set(self.hard_negatives[qid]["pos"])
            neg = set()
            # only use top-{lower_bound} negatives
            neg_list = [
                neg_docs[self.skip_top_n:self.use_top_n] for neg_docs in self.hard_negatives[qid]["model_negatives"].values()]

            assert all(
                [len(neg_docs) <= self.use_top_n for neg_docs in neg_list])

            if sample_mode == "bottom":
                for _ in range(neg_per_model):
                    for i in range(len(neg_list)):
                        last_doc = neg_list[i].pop()
                        while last_doc in pos or last_doc in neg:
                            last_doc = neg_list[i].pop()
                        neg.add(last_doc)

            elif sample_mode == "random":
                for _ in range(neg_per_model):
                    for i in range(len(neg_list)):
                        
                        if len(neg_list[i]) == 0:
                            continue

                        random_doc = random.choice(neg_list[i])
                        neg_list[i].remove(random_doc)
                        
                        while ((random_doc in pos) or (random_doc in neg)):
                            if len(neg_list[i]) == 0:
                                break
                            random_doc = random.choice(neg_list[i])
                            neg_list[i].remove(random_doc)
                            
                        neg.add(random_doc)

            if len(pos) == 0 or len(neg) == 0:
                continue
            
            neg.difference_update(pos)
            
            for neg_id in neg:
                self.train_pairs.append(
                    [qid, neg_id, neg_label])
                neg_counts += 1

            selected_pos_doc_ids = random.choices(
                list(pos), k=len(neg) // pos_neg_ration)

            for pos_id in selected_pos_doc_ids:
                self.train_pairs.append(
                    [qid, pos_id, pos_label])
                pos_counts += 1

        self.check_train_pairs(self.train_pairs)

        logging.info(f"Total {len(self.train_pairs)} train samples")
        logging.info(f"Pos counts: {pos_counts} Neg counts: {neg_counts}")

    def create_train_samples(self, neg_per_query, pos_neg_ration, pos_label=1, neg_label=0):

        logging.info(
            f"Creating train samples with {neg_per_query} negative samples per query and {pos_neg_ration} pos/neg ratio")

        if self.hard_negatives is None:
            self.hard_negatives = self.load_hard_negatives(
                os.path.join(self.generated_path, self.hard_negatives_name), self.neg_retrievers)

        self.train_pairs = []
        pos_counts, neg_counts = 0, 0

        for qid in self.hard_negatives:

            pos_pool = self.hard_negatives[qid]["pos"]
            neg_pool = set()

            for neg_ids in self.hard_negatives[qid]["model_negatives"].values():
                neg_pool.update(neg_ids[self.skip_top_n:self.use_top_n])

            # remove pos from neg
            neg_pool.difference_update(pos_pool)

            if len(pos_pool) == 0 or len(neg_pool) == 0:
                continue

            selected_neg_doc_ids = random.sample(
                list(neg_pool), min(neg_per_query, len(neg_pool)))
            selected_pos_doc_ids = random.choices(
                list(pos_pool), k=len(selected_neg_doc_ids) // pos_neg_ration)

            for neg_id in selected_neg_doc_ids:
                self.train_pairs.append(
                    [qid, neg_id, neg_label]
                )

            for pos_id in selected_pos_doc_ids:
                self.train_pairs.append(
                    [qid, pos_id, pos_label]
                )

            pos_counts += len(selected_pos_doc_ids)
            neg_counts += len(selected_neg_doc_ids)

        self.check_train_pairs(self.train_pairs)
        logging.info(f"Total {len(self.train_pairs)} train samples")
        logging.info(f"Pos counts: {pos_counts} Neg counts: {neg_counts}")

    def __len__(self):
        return len(self.train_pairs)

    def __getitem__(self, index):

        qid, doc_id, label = self.train_pairs[index]
        query_text = self.gen_queries[str(qid)]
        doc_text = self.get_doc_text(doc_id)
        return InputExample(texts=[query_text, doc_text],
                            label=float(label))
