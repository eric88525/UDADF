import os
import csv
from typing import List, Tuple
from beir import util
from beir.datasets.data_loader import GenericDataLoader
from sentence_transformers.cross_encoder import CrossEncoder
import logging
from beir import util, LoggingHandler
import numpy as np
from collections import Counter
import math

logger = logging.getLogger(__name__)

class CrossFilter:

    def __init__(
        self,
        corpus,
        queries,
    ) -> None:

        self.corpus, self.queries = corpus, queries

    @staticmethod
    def load_pseudo_qrels(rel_file: str, skip_header=True, load_score=False):

        logging.info(f"Loading pseudo qrels from {rel_file}")

        pseudo_qrels = []
        with open(rel_file, 'r') as f:
            reader = csv.reader(f, delimiter='\t')

            if skip_header:
                next(reader)

            for line in reader:
                if load_score:
                    assert len(
                        line
                    ) == 3, "pseudo_qrels should be a list of [query_id, doc_id, score]"
                    line[-1] = float(line[-1])
                    pseudo_qrels.append(line)
                else:
                    pseudo_qrels.append(line[:2])  # only query and doc id

        return pseudo_qrels

    def filter_first_word(self,
                          pseudo_qrels: List[Tuple[str, str]],
                          extra_first_words: List[str] = []):
        """ Filter the pseudo qrels based on the query

        Args:
            pseudo_qrels (List[Tuple[str, str]]): list of [query_id, doc_id]
        """
        logging.info(
            f"Filtering {len(pseudo_qrels)} pseudo qrels based on first word of query"
        )

        first_words = [
            'what', 'why', 'which', 'who', 'when', 'how', 'are', 'is', 'does'
        ] + extra_first_words
        pseudo_qrels = [
            qrel for qrel in pseudo_qrels
            if qrel[0] in self.queries and self.queries[qrel[0]] != ''
            and self.queries[qrel[0]].lower().split()[0] in first_words
        ]

        logging.info(
            f"Filtered {len(pseudo_qrels)} pseudo qrels based on first word of query"
        )

        return pseudo_qrels

    def score_pseudo_qrels(self,
                           pseudo_qrels: List[Tuple[str, str]],
                           model_name: str,
                           device: str = 'cuda:0',
                           max_length=350,
                           batch_size: int = 1024):

        logging.info(f"Scoring pseudo qrels")

        model = CrossEncoder(model_name=model_name,
                             device=device,
                             max_length=max_length)

        logging.info(f"Predicting scores for {len(pseudo_qrels)} pseudo qrels")
        pred_scores = model.predict(
            sentences=[[
                self.queries[qid],
                self.corpus[pid]["title"] + self.corpus[pid]["text"]
            ] for qid, pid in pseudo_qrels],
            show_progress_bar=True,
            batch_size=batch_size,
        )

        score_qrels = [[qid, pid, score]
                       for (qid, pid), score in zip(pseudo_qrels, pred_scores)]

        return score_qrels

    @staticmethod
    def save_tsv(pseudo_qrels_score, output_path, sort=True):

        assert len(
            pseudo_qrels_score[0]
        ) == 3, "pseudo_qrels_score should be a list of [query_id, doc_id, score]"

        if sort:
            pseudo_qrels_score = sorted(pseudo_qrels_score,
                                        key=lambda x: float(x[2]),
                                        reverse=True)

        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

        logging.info(f"Saving pseudo qrels to {output_path}")

        with open(output_path, 'w') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(['query_id', 'doc_id', 'score'])
            writer.writerows(pseudo_qrels_score)

    @staticmethod
    def get_top_n(pseudo_qrels_score, n=100000):

        assert len(
            pseudo_qrels_score[0]
        ) == 3, "pseudo_qrels_score should be a list of [query_id, doc_id, score]"

        pseudo_qrels_score = sorted(pseudo_qrels_score,
                                    key=lambda x: float(x[2]),
                                    reverse=True)
        pseudo_qrels_score = pseudo_qrels_score[:n]

        return pseudo_qrels_score

    @staticmethod
    def set_score(pseudo_qrels_score, score=1):
        return [[qid, pid, score] for qid, pid, _ in pseudo_qrels_score]

    def remove_empty_queries(self, pseudo_qrels):

        result = [qrel for qrel in pseudo_qrels if qrel[0]
                  in self.queries and self.queries[qrel[0]] != '']
        logging.info(
            f"Removed {len(pseudo_qrels) - len(result)} pseudo qrels with empty queries")

        return result

    def remove_short_queries(self, pseudo_qrels, min_length=5):

        result = [qrel for qrel in pseudo_qrels
                  if self.queries[qrel[0]] != '' and len(self.queries[qrel[0]].split()) >= min_length]

        logging.info(
            f"Removed {len(pseudo_qrels) - len(result)} pseudo qrels with short queries")

        return result

    def keep_topn_common_first_words(self, pseudo_qrels, top_n=50):

        logging.info(f"Keeping top {top_n} common first words")

        first_words = []

        for qrel in pseudo_qrels:
            try:
                first_words.append(self.queries[qrel[0]].lower().split()[0])
            except:
                ...
        first_words = [str(i[0]) for i in Counter(first_words).most_common(top_n)]
        logging.info(f"Top-50 first words: {first_words}")
        first_words = set(first_words)

        result = [qrel for qrel in pseudo_qrels if self.queries[qrel[0]].lower().split()[
            0] in first_words]

        logging.info(
            f"Removed {len(pseudo_qrels) - len(result)} pseudo qrels with uncommon first words")

        return result
    @staticmethod
    def remove_low_score(pseudo_qrels_score, score, sigmoid=False):
        logging.info(f"Removing pseudo qrels with score lower than {score}")
        assert all([len(qrel) == 3 for qrel in pseudo_qrels_score]), "pseudo_qrels_score should be a list of [query_id, doc_id, score]"
        logging.info(f"Before: {len(pseudo_qrels_score)}")

        if sigmoid:
            pseudo_qrels_score = [[qid, pid, 1/(1+math.exp(-float(s)))] for qid, pid, s in pseudo_qrels_score]

        pseudo_qrels_score = [qrel for qrel in pseudo_qrels_score if float(qrel[2]) >= score]
        logging.info(f"After: {len(pseudo_qrels_score)}")
        return pseudo_qrels_score