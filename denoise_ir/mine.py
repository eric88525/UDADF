import json
from beir.datasets.data_loader import GenericDataLoader
from sentence_transformers import SentenceTransformer
import torch
from easy_elasticsearch import ElasticSearchBM25
import tqdm
import numpy as np
import os
import logging
import argparse
import time

logger = logging.getLogger(__name__)


class NegativeMiner(object):

    def __init__(
        self,
        generated_path,
        prefix="",
        split="train",
        corpus_file: str = "corpus.jsonl",
        query_file: str = "queries.jsonl",
        qrels_folder: str = "qrels",
        retrievers=[
            "bm25", "msmarco-distilbert-base-v3", "msmarco-MiniLM-L-6-v3"
        ],
        retriever_score_functions=["none", "cos_sim", "cos_sim"],
        nneg=50,
        use_train_qrels: bool = False,
        batch_size=128,
        device: str = "cuda",
        bm25_index=None,
        convert_to_numpy=True,
    ):
        self.batch_size = batch_size
        self.device = device
        self.bm25_index = bm25_index
        self.convert_to_numpy = convert_to_numpy

        if use_train_qrels:
            logger.info(
                "Using labeled qrels to construct the hard-negative data")
            self.corpus, self.gen_queries, self.gen_qrels = GenericDataLoader(
                generated_path).load(split="train")
        else:
            self.corpus, self.gen_queries, self.gen_qrels = GenericDataLoader(
                generated_path,
                prefix=prefix,
                corpus_file=corpus_file,
                query_file=query_file,
                qrels_folder=qrels_folder,
            ).load(split=split)

        self.retrievers = retrievers
        self.retriever_score_functions = retriever_score_functions
        if "bm25" in retrievers:
            assert (
                nneg <= 10000
            ), "Only `negatives_per_query` <= 10000 is acceptable by Elasticsearch-BM25"
            assert retriever_score_functions[retrievers.index(
                "bm25")] == "none"

        assert set(retriever_score_functions).issubset(
            {"none", "dot", "cos_sim"})

        self.nneg = nneg
        if nneg > len(self.corpus):
            logger.warning(
                "`negatives_per_query` > corpus size. Please use a smaller `negatives_per_query`"
            )
            self.nneg = len(self.corpus)

    def _get_doc(self, did):
        return " ".join([self.corpus[did]["title"], self.corpus[did]["text"]])

    def _mine_sbert(self, model_name, score_function):
        logger.info(f"Mining with {model_name}")
        assert score_function in ["dot", "cos_sim"]
        normalize_embeddings = False
        if score_function == "cos_sim":
            normalize_embeddings = True

        result = {}
        sbert = SentenceTransformer(model_name, device=self.device)
        docs = list(map(self._get_doc, self.corpus.keys()))
        dids = np.array(list(self.corpus.keys()))
        doc_embs = sbert.encode(
            docs,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=self.convert_to_numpy,
            convert_to_tensor=not self.convert_to_numpy,
            normalize_embeddings=normalize_embeddings,
        )
        qids = list(self.gen_qrels.keys())
        queries = list(map(lambda qid: self.gen_queries[qid], qids))
        for start in tqdm.trange(0, len(queries), self.batch_size):
            qid_batch = qids[start:start + self.batch_size]
            qemb_batch = sbert.encode(
                queries[start:start + self.batch_size],
                show_progress_bar=False,
                convert_to_numpy=self.convert_to_numpy,
                convert_to_tensor=not self.convert_to_numpy,
                normalize_embeddings=normalize_embeddings,
            )
            if self.convert_to_numpy:
                score_mtrx = np.dot(qemb_batch, doc_embs.T)
                score_mtrx = torch.from_numpy(score_mtrx)  # (qsize, dsize)
            else:
                score_mtrx = torch.matmul(qemb_batch,
                                          doc_embs.t())  # (qsize, dsize)
            _, indices_topk = score_mtrx.topk(k=self.nneg, dim=-1)
            indices_topk = indices_topk.tolist()
            for qid, neg_dids in zip(qid_batch, indices_topk):
                neg_dids = dids[neg_dids].tolist()
                for pos_did in self.gen_qrels[qid]:
                    if pos_did in neg_dids:
                        neg_dids.remove(pos_did)
                result[qid] = neg_dids
        return result

    def _mine_bm25(self):
        logger.info(f"Mining with bm25")
        result = {}
        docs = list(map(self._get_doc, self.corpus.keys()))
        dids = np.array(list(self.corpus.keys()))
        pool = dict(zip(dids, docs))
        bm25 = ElasticSearchBM25(
            pool,
            port_http="9222",
            port_tcp="9333",
            service_type="executable",
            index_name=f"one_trial{int(time.time() * 1000000)}",
        )
        for qid, pos_dids in tqdm.tqdm(self.gen_qrels.items()):
            query = self.gen_queries[qid]
            rank = bm25.query(query, topk=self.nneg)  # topk should be <= 10000
            neg_dids = list(rank.keys())
            for pos_did in self.gen_qrels[qid]:
                if pos_did in neg_dids:
                    neg_dids.remove(pos_did)
            result[qid] = neg_dids
        return result

    def pyserini_bm25(self):
        from pyserini.search.lucene import LuceneSearcher

        logger.info(f"Mining with bm25, using {self.bm25_index} index")

        bm25 = LuceneSearcher.from_prebuilt_index(self.bm25_index)
        bm25.set_bm25(0.9, 0.4)
        bm25_hits = {}

        def hit_template(hits):
            results = {}
            for qid, hit in hits.items():
                results[qid] = [h.docid for h in hit]
            return results

        qids = list(self.gen_qrels.keys())
        logger.info(f"Total number of queries: {len(qids)}")

        for i in tqdm.tqdm(range(0, len(qids), 100), ncols=50):
            qids_batch = qids[i:i+100]
            qids_text_batch = [self.gen_queries[qid] for qid in qids_batch]
            hits = bm25.batch_search(  # hits is a dict of {query_id: [hits]}
                queries=qids_text_batch,
                qids=qids_batch,
                k=self.nneg,
                threads=8,
            )
            bm25_hits.update(hit_template(hits))

        for qid, pos_dids in tqdm.tqdm(self.gen_qrels.items()):
            bm25_hits[qid] = [
                did for did in bm25_hits[qid] if did not in pos_dids]

        return bm25_hits

    def run(self, output_path):

        if os.path.exists(output_path):
            y = input(f"{output_path} already exists. Overwrite? [y/n]")

            if y.lower() != "y":
                logger.info("Exiting")
                return

        hard_negatives = {}
        for retriever, score_function in zip(self.retrievers,
                                             self.retriever_score_functions):
            if retriever == "bm25":
                hard_negatives[retriever] = self._mine_bm25(
                ) if self.bm25_index is None else self.pyserini_bm25()
            else:
                hard_negatives[retriever] = self._mine_sbert(
                    retriever, score_function)

        logger.info("Combining all the data")
        result_jsonl = []
        for qid, pos_dids in tqdm.tqdm(self.gen_qrels.items()):
            line = {
                "qid": qid,
                "pos": list(pos_dids.keys()),
                "neg": {k: v[qid]
                        for k, v in hard_negatives.items()},
            }
            result_jsonl.append(line)

        logger.info(f"Saving data to {output_path}")
        with open(output_path, "w") as f:
            for line in result_jsonl:
                f.write(json.dumps(line) + "\n")
        logger.info("Done")
        return result_jsonl