import json
import os
import pickle
import logging
from beir import util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.reranking import Rerank
from typing import Any, List, Dict, Tuple
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
import os
import logging
logger = logging.getLogger(__name__)


class TestEvaluator:
    def __init__(self,
                 retrieval_result,
                 dataset,
                 base_path=os.getcwd(),
                 batch_size=512,
                 topk=[100, 1000],
                 k_values=[5, 10, 100, 200, 500, 1000],
                 device="cuda:0",
                 max_seq_length=300,
                 split="test",
                 test_matrix="NDCG@10",
                 max_test_samples=100):

        self.base_path = base_path
        self.batch_size = batch_size

        if not os.path.exists(retrieval_result):
            raise FileNotFoundError(
                f"Retrieval results not found at {self.retrieval_result}")
        else:
            self.load_retrieval_result(retrieval_result, max_test_samples)
            
        self.topk = topk
        self.dataset = dataset
        self.k_values = k_values
        self.device = device
        self.max_seq_length = max_seq_length
        self.split = split
        self.test_matrix = test_matrix
        self.max_test_samples = max_test_samples

        url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(
            self.dataset)
        out_dir = os.path.join(self.base_path, "datasets")
        data_path = util.download_and_unzip(url, out_dir)
        self.corpus, self.queries, self.qrels = GenericDataLoader(
            data_path).load(split=self.split)

    def load_retrieval_result(self, retrieval_result: str, max_test_samples: int) -> Dict[str, List[Tuple[str, float]]]:
        
        if retrieval_result.endswith(".pkl"):
            with open(retrieval_result, "rb") as f:
                self.to_rerank = pickle.load(f)
        elif retrieval_result.endswith(".json"):
            with open(retrieval_result, "r") as f:
                self.to_rerank = json.load(f)

        if len(self.to_rerank) > max_test_samples:
            qids = list(self.to_rerank.keys())
            qids = sorted(qids)[:max_test_samples]
            self.to_rerank = {qid: self.to_rerank[qid] for qid in qids}

        logging.info(f"Loaded {len(self.to_rerank)} retrieval results")


    def __call__(self, model, *args: Any, **kwds: Any) -> Any:

        logging.info(f"Running evaluation on {self.dataset} dataset")
        self.reranker = Rerank(model, batch_size=self.batch_size)
        report = []
        for k in self.topk:

            results = self.reranker.rerank(
                corpus=self.corpus,
                queries=self.queries,
                results=self.to_rerank,
                top_k=k
            )

            ndcg, _map, recall, precision = EvaluateRetrieval.evaluate(
                qrels=self.qrels,
                results=results,
                k_values=self.k_values,
            )

            result = {}
            result.update(ndcg)
            result.update(_map)
            result.update(recall)
            result.update(precision)

            report.append({
                "test_matrix": f"Top@{k}-{self.test_matrix}",
                "score": result[self.test_matrix]
            })
        print(report)
        return report
