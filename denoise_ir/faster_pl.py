import logging
import os
import json
import random
from collections import defaultdict
import csv
from typing import List
from .model import DenoiseCrossEncoder
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import numpy as np

logger = logging.getLogger(__name__)

class HardNegDataset(Dataset):
    def __init__(self, jsonl_path, range_start=1, range_end=50):

        # list of {"qid": qid, "pos": [doc_id, doc_id...], "neg": [doc_id, doc_id...]}
        self.hard_negatives = self.load_hard_negatives(
            jsonl_path, range_start, range_end)

        self.history = set()
        print("len of hard_negatives", len(self.hard_negatives))

    def __len__(self):
        return len(self.hard_negatives)

    def __getitem__(self, idx):
        item = self.hard_negatives[idx]
        qid = item["qid"]
        neg_id = random.choice(item["neg"])
        pos_id = random.choice(item["pos"])

        while f"{qid}_{pos_id}_{neg_id}" in self.history:
            neg_id = random.choice(item["neg"])

        self.history.add(f"{qid}_{pos_id}_{neg_id}")
        return qid, pos_id, neg_id

    @staticmethod
    def load_hard_negatives(hard_negatives_path, range_start=1, range_end=50):

        if not os.path.exists(hard_negatives_path):
            raise FileNotFoundError(
                f"Hard negatives path {hard_negatives_path} does not exist")

        hard_negatives = defaultdict(dict)

        with open(hard_negatives_path, "r") as f:
            for line in f:
                # row = {'qid': str, 'pos': list, 'neg': dict of {model_name: list}}
                row = json.loads(line)
                if row["qid"] not in hard_negatives:
                    hard_negatives[row["qid"]] = {
                        "qid": row["qid"],
                        "pos": set(),
                        "neg": set(),
                    }
                hard_negatives[row["qid"]]["pos"].update(row["pos"])

                for negatives in row["neg"].values():
                    hard_negatives[row["qid"]]["neg"].update(
                        negatives[max(0, range_start - 1): range_end]
                    )

                # remove positives from negative pools
                hard_negatives[row["qid"]]["neg"].difference_update(
                    hard_negatives[row["qid"]]["pos"])

        for k in hard_negatives.keys():
            hard_negatives[k]["pos"] = list(hard_negatives[k]["pos"])
            hard_negatives[k]["neg"] = list(hard_negatives[k]["neg"])

        return list(hard_negatives.values())

    @staticmethod
    def collate_fn(batch):
        qid = [b[0] for b in batch]
        pos = [b[1] for b in batch]
        neg = [b[2] for b in batch]
        return qid, pos, neg

class PseudoLabeler(object):
    def __init__(
        self,
        generated_path,
        gen_queries: dict,
        corpus: dict,
        total_steps: int,
        batch_size: int,
        cross_encoders: List[str],
        max_seq_length: int,
        hard_negatives="hard-negatives.jsonl",
        output_file_name="gpl-training-data.tsv",
        cross_encoder_batch_size=32,
        device="cuda",
    ):
        self.cross_encoder_batch_size = cross_encoder_batch_size
        self.fpath_hard_negatives = os.path.join(
            generated_path, hard_negatives)

        self.cross_encoders = []
        for ce in cross_encoders:
            self.cross_encoders.append(
                DenoiseCrossEncoder([ce], device=device, max_length=max_seq_length)
            )
        self.output_path = os.path.join(generated_path,
                                        output_file_name)
        self.batch_size = batch_size
        self.total_steps = total_steps
        self.max_seq_length = max_seq_length
        self.corpus = corpus
        self.gen_queries = gen_queries

    @staticmethod
    def build_train_data(fpath_hard_negatives, batch_size, total_steps, range_start=1, range_end=50):

        logger.info("Sampling (qid, pos, neg) pairs")
        hard_negative_dataset = HardNegDataset(
            fpath_hard_negatives, range_start=range_start, range_end=range_end)

        loader = DataLoader(hard_negative_dataset,
                            shuffle=True,
                            batch_size=batch_size,
                            drop_last=True,
                            collate_fn=HardNegDataset.collate_fn)

        sample_iter = iter(loader)

        qid_list, pos_list, neg_list = [], [], []

        for _ in tqdm(range(total_steps), desc="Sampling (qid, pos, neg) pairs"):
            try:
                qid, pos, neg = next(sample_iter)
            except StopIteration:
                sample_iter = iter(loader)
                qid, pos, neg = next(sample_iter)

            qid_list.extend(qid)
            pos_list.extend(pos)
            neg_list.extend(neg)

        print(f"size of qid_list={len(qid_list)}")
        print(f"size of pos_list={len(pos_list)}")
        print(f"size of neg_list={len(neg_list)}")

        # check not duplicates q, pos, neg
        his = set()
        for q, p, n in zip(qid_list, pos_list, neg_list):
            his.add(f"{q}_{p}_{n}")

        assert len(his) == total_steps * batch_size, f"size of his={len(his)}"

        return qid_list, pos_list, neg_list

    def get_doc(self, doc_id):
        return self.corpus[doc_id].get("title", "") + " " + self.corpus[doc_id].get("text", "")

    def create_batch(self, qid_list, pos_list, neg_list, device_ids=[0]):

        q_texts = [self.gen_queries[qid] for qid in qid_list]
        pos_texts = [self.get_doc(doc_id) for doc_id in pos_list]
        neg_texts = [self.get_doc(doc_id) for doc_id in neg_list]

        avg_margins = []

        for ce in self.cross_encoders:
            # scores = relation of (q, pos) + (q, neg)
            scores = ce.parallel_predict(
                list(zip(q_texts, pos_texts)) + list(zip(q_texts, neg_texts)),
                batch_size=self.cross_encoder_batch_size,
                show_progress_bar=False,
                device_ids=device_ids,
                convert_to_numpy=True,
            )
            # margin is CE(q, pos) - CE(q, neg)
            margin = scores[:len(qid_list)] - scores[len(qid_list):]
            avg_margins.append(margin)

        avg_margins = np.mean(avg_margins, axis=0)
        avg_margins = avg_margins.tolist()

        batch_distill = map(
            lambda quad: "\t".join((*quad[:3], str(quad[3]))) + "\n",
            zip(qid_list, pos_list, neg_list, avg_margins),
        )
        return batch_distill

    def run(self, device_ids=[0], to_relabel=None):

        logger.info("============================================")
        logger.info("Start pseudo labeling")
        logger.info(f"Output file: [{self.output_path}]")
        logger.info("============================================")

        if to_relabel and os.path.exists(to_relabel):

            with open(to_relabel, "r") as file:

                qid_list, pos_list, neg_list = [], [], []
                reader = csv.reader(file, delimiter='\t')
                rows = [row for row in reader]
                qid_list = [r[0] for r in rows]
                pos_list = [r[1] for r in rows]
                neg_list = [r[2] for r in rows]
                logger.info(f"Load {to_relabel}, len: {len(qid_list)}")
        else:
            qid_list, pos_list, neg_list = self.build_train_data(
                self.fpath_hard_negatives, self.batch_size, self.total_steps)

        unique = set(zip(qid_list, pos_list, neg_list))
        logger.info(f"Unique pairs: {len(unique)}")

        if to_relabel is None:
            assert len(unique) == self.total_steps * \
                self.batch_size, "Duplicates in (qid, pos, neg) pairs"

        logger.info(f"Data pairs {len(qid_list)}")
        
        # chunk size = 1000
        chunk_size = 2048
        for i in tqdm(range(0, len(qid_list), chunk_size)):
            batch_distill = self.create_batch(
                qid_list[i:i + chunk_size], pos_list[i:i + chunk_size], neg_list[i:i + chunk_size], device_ids)
            with open(self.output_path, "a") as f:
                f.writelines(batch_distill)

        logger.info(f"Saved distill data to {self.output_path}")
