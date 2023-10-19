from sentence_transformers import SentenceTransformer
import sentence_transformers.models as sent_models
import logging
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
import sentence_transformers
from beir.retrieval import models
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from sentence_transformers import SentenceTransformer
import sentence_transformers
import os
import logging
import numpy as np
import json
from typing import List
import re
import argparse

logger = logging.getLogger(__name__)

def directly_loadable_by_sbert(model: SentenceTransformer):
    loadable_by_sbert = True
    try:
        texts = [
            "This is an input text",
        ]
        model.encode(texts)
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            raise e
        else:
            loadable_by_sbert = False
    return loadable_by_sbert


def load_sbert(model_name_or_path, pooling=None, max_seq_length=None):
    model = SentenceTransformer(model_name_or_path)

    ## Check whether SBERT can load the checkpoint and use it
    loadable_by_sbert = directly_loadable_by_sbert(model)
    if loadable_by_sbert:
        ## Loadable by SBERT directly
        ## Mainly two cases: (1) The checkpoint is in SBERT-format (e.g. "bert-base-nli-mean-tokens"); (2) it is in HF-format but the last layer can provide a hidden state for each token (e.g. "bert-base-uncased")
        ## NOTICE: Even for (2), there might be some checkpoints (e.g. "princeton-nlp/sup-simcse-bert-base-uncased") that uses a linear layer on top of the CLS token embedding to get the final dense representation. In this case, setting `--pooling` to a specify pooling method will misuse the checkpoint. This is why we recommend to use SBERT-format if possible
        ## Setting pooling if needed
        if pooling is not None:
            logger.warning(
                f"Trying setting pooling method manually (`--pooling={pooling}`). Recommand to use a checkpoint in SBERT-format and leave the `--pooling=None`: This is less likely to misuse the pooling"
            )
            last_layer: sent_models.Pooling = model[-1]
            assert (
                type(last_layer) == sent_models.Pooling
            ), f"The last layer is not a pooling layer and thus `--pooling={pooling}` cannot work. Please try leaving `--pooling=None` as in the default setting"
            # We here change the pooling by building the whole SBERT model again, which is safer and more maintainable than setting the attributes of the Pooling module
            word_embedding_model = sent_models.Transformer(
                model_name_or_path, max_seq_length=max_seq_length
            )
            pooling_model = sent_models.Pooling(
                word_embedding_model.get_word_embedding_dimension(),
                pooling_mode=pooling,
            )
            model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    else:
        ## Not directly loadable by SBERT
        ## Mainly one case: The last layer is a linear layer (e.g. "facebook/dpr-question_encoder-single-nq-base")
        raise NotImplementedError(
            "This checkpoint cannot be directly loadable by SBERT. Please transform it into SBERT-format first and then try again. Please pay attention to its last layer"
        )

    ## Setting max_seq_length if needed
    if max_seq_length is not None:
        first_layer: sent_models.Transformer = model[0]
        assert (
            type(first_layer) == sent_models.Transformer
        ), "Unknown error, please report this"
        assert hasattr(
            first_layer, "max_seq_length"
        ), "Unknown error, please report this"
        setattr(
            first_layer, "max_seq_length", max_seq_length
        )  # Set the maximum-sequence length
        logger.info(f"Set max_seq_length={max_seq_length}")

    return model








logger = logging.getLogger(__name__)


def evaluate(
    data_path: str,
    output_dir: str,
    model_name_or_path: str,
    max_seq_length: int = 350,
    score_function: str = "dot",
    pooling: str = None,
    sep: str = " ",
    k_values: List[int] = [1, 3, 5, 10, 20, 100],
    split: str = "test",
    batch_size: int = 128,
):

    model: SentenceTransformer = load_sbert(model_name_or_path, pooling, max_seq_length)

    pooling_module: sentence_transformers.models.Pooling = model._last_module()
    assert type(pooling_module) == sentence_transformers.models.Pooling
    pooling_mode = pooling_module.get_pooling_mode_str()
    logger.info(
        f"Running evaluation with setting: max_seq_length = {max_seq_length}, score_function = {score_function}, split = {split} and pooling: {pooling_mode}"
    )

    data_paths = []
    if "cqadupstack" in data_path:
        data_paths = [
            os.path.join(data_path, sub_dataset)
            for sub_dataset in [
                "android",
                "english",
                "gaming",
                "gis",
                "mathematica",
                "physics",
                "programmers",
                "stats",
                "tex",
                "unix",
                "webmasters",
                "wordpress",
            ]
        ]
    else:
        data_paths.append(data_path)

    ndcgs = []
    _maps = []
    recalls = []
    precisions = []
    mrrs = []
    for data_path in data_paths:
        try:
            corpus, queries, qrels = GenericDataLoader(data_path).load(split=split)
        except ValueError as e:
            missing_files = re.search(
                r"File (.*) not present! Please provide accurate file.", str(e)
            )
            if missing_files:
                raise ValueError(
                    f"Missing evaluation data files ({missing_files.groups()}). "
                    f"Please put them under {data_path} or set `do_evaluation`=False."
                )
            else:
                raise e

        sbert = models.SentenceBERT(sep=sep)
        sbert.q_model = model
        sbert.doc_model = model

        model_dres = DRES(sbert, batch_size=batch_size)
        assert score_function in ["dot", "cos_sim"]
        retriever = EvaluateRetrieval(
            model_dres, score_function=score_function, k_values=k_values
        )  # or "dot" for dot-product
        results = retriever.retrieve(corpus, queries)

        #### Evaluate your retrieval using NDCG@k, MAP@K ...
        ndcg, _map, recall, precision = EvaluateRetrieval.evaluate(
            qrels, results, k_values
        )
        mrr = EvaluateRetrieval.evaluate_custom(qrels, results, k_values, metric="mrr")
        ndcgs.append(ndcg)
        _maps.append(_map)
        recalls.append(recall)
        precisions.append(precision)
        mrrs.append(mrr)

    ndcg = {k: np.mean([score[k] for score in ndcgs]) for k in ndcg}
    _map = {k: np.mean([score[k] for score in _maps]) for k in _map}
    recall = {k: np.mean([score[k] for score in recalls]) for k in recall}
    precision = {k: np.mean([score[k] for score in precisions]) for k in precision}
    mrr = {k: np.mean([score[k] for score in mrrs]) for k in mrr}

    os.makedirs(output_dir, exist_ok=True)
    result_path = os.path.join(output_dir, "results.json")
    with open(result_path, "w") as f:
        json.dump(
            {
                "ndcg": ndcg,
                "map": _map,
                "recall": recall,
                "precicion": precision,
                "mrr": mrr,
            },
            f,
            indent=4,
        )
    logger.info(f"Saved evaluation results to {result_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path")
    parser.add_argument("--output_dir")
    parser.add_argument("--model_name_or_path")
    parser.add_argument("--max_seq_length", type=int, default=350)
    parser.add_argument("--score_function", choices=["dot", "cos_sim"], default="dot")
    parser.add_argument("--pooling", choices=["mean", "cls", "max"], default=None)
    parser.add_argument(
        "--sep",
        type=str,
        default=" ",
        help="Separation token between title and body text for each passage. The concatenation way is `sep.join([title, body])`",
    )
    parser.add_argument(
        "--k_values",
        nargs="+",
        type=int,
        default=[1, 3, 5, 10, 20, 100],
        help="The K values in the evaluation. This will compute nDCG@K, recall@K, precision@K and MAP@K",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "test", "dev"],
        help="Which split to evaluate on",
    )
    args = parser.parse_args()
    evaluate(**vars(args))
