from denoise_ir.mine import NegativeMiner
from denoise_ir.filter import CrossFilter
import logging
import math
import json
import os
from beir.datasets.data_loader import GenericDataLoader
from beir.generation.models import QGenModel
from beir.generation import QueryGenerator as QGen
from beir import util
import csv
import random
from .utils import prefix_join, load_corpus, load_queries

# Mapping for retriever functions
retriever_mapping = {
    "bm25": "none",
    "msmarco-distilbert-base-v3": "cos_sim",
    "msmarco-MiniLM-L-6-v3": "cos_sim",
}

def resize(data_path, output_path, new_size, use_qrels=[]):
    """
    Resizes the corpus in the given data path to the specified new size and saves it in the output path.

    Args:
        data_path (str): Path to the original data.
        output_path (str): Path to save the resized corpus.
        new_size (int): New size of the corpus.
        use_qrels (list, optional): List of query relevance files to expand the corpus. Defaults to [].
    """
    logging.info(f"Resizing the corpus in {data_path} to {output_path} with new size {new_size}")

    # Load the corpus
    corpus = load_corpus(data_path, remove_empty=True)
    logging.info(f"Corpus size: {len(corpus)} (remove empty documents)")

    if len(corpus) <= new_size:
        logging.warning("`new_size` should be smaller than the corpus size")
        corpus_new = list(corpus.items())
    else:
        logging.info(f"Sampling {new_size} documents from the corpus")
        corpus_new = random.sample(list(corpus.items()), k=new_size)

    corpus_new = dict(corpus_new)
    for split in use_qrels:
        # Load the query relevance file
        qrels_path = os.path.join(data_path, "qrels", f"{split}.tsv")
        if not os.path.exists(qrels_path):
            logging.warning(f"{split} qrels not found in {qrels_path}. Skipping")
        else:
            qrels = CrossFilter.load_pseudo_qrels(qrels_path, load_score=False)  # list of [qid, pid]
            docs = set([str(item[-1]) for item in qrels])

            for doc_id in docs:
                if doc_id not in corpus_new and doc_id in corpus:
                    corpus_new[doc_id] = corpus[doc_id]
            logging.info(f"Expanded {split}, corpus size: {len(corpus_new)}")

    # Create the output path
    os.makedirs(output_path, exist_ok=True)
    with open(os.path.join(output_path, "corpus.jsonl"), "w") as f:
        for doc_id, doc in corpus_new.items():
            doc["_id"] = doc_id
            f.write(json.dumps(doc) + "\n")

    logging.info(f"Resized the corpus in {data_path} to {output_path} with new size {new_size}")

def prefix_join(prefix, name):
    """
    Joins the given prefix and name.

    Args:
        prefix (str): Prefix string.
        name (str): Name string.

    Returns:
        str: Joined string.
    """
    if prefix:
        return prefix + "-" + name
    else:
        return name


def download_dataset(dataset: str, base_path: str):
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(
        dataset)
    out_dir = os.path.join(base_path, "datasets")
    data_folder = util.download_and_unzip(url, out_dir)
    return data_folder

def generate_pseudo_queries(
        pseudo_data_folder: str,
        prefix: str = "gen",
        batch_size: int = 10,
        model_path: str = "BeIR/query-gen-msmarco-t5-base-v1",
        device="cuda:0"):
    """Generates pseudo queries using a pre-trained query generation model.

    Args:
        pseudo_data_folder (str): Path to the folder containing the data.
        prefix (str, optional): Prefix for the output files. Defaults to "".
        batch_size (int, optional): Batch size for the query generator. Defaults to 10.
        model_path (str, optional): Path to the query generator model. Defaults to "BeIR/query-gen-msmarco-t5-large-v1".
    """
    logging.info("Generating pseudo queries")

    generator = QGen(model=QGenModel(model_path, device=device))

    corpus = load_corpus(pseudo_data_folder)
    if len(corpus) * 3 < 3e5:
        ques_per_passage = math.ceil(3e5 / len(corpus))
    else:
        ques_per_passage = 3

    generator.generate(
        corpus,
        output_dir=pseudo_data_folder,
        ques_per_passage=ques_per_passage,
        prefix=prefix,
        batch_size=batch_size,
    )


def score_pseudo_qrels(
    pseudo_data_folder: str,
    prefix: str = "gen",
    cross_config: dict = {
        "model_name": "cross-encoder/ms-marco-MiniLM-L-12-v2",
        "device": "cuda:0",
        "batch_size": 1024,
        "max_length": 350
    },
    target_prefix=None,
):
    """Scores the pseudo qrels using a cross-encoder."""
    logging.info(
        f"Scoring pseudo qrels using {cross_config['model_name']}")

    corpus = load_corpus(pseudo_data_folder)
    queries = load_queries(pseudo_data_folder, prefix)

    filter = CrossFilter(corpus=corpus, queries=queries)

    rel_file = os.path.join(pseudo_data_folder, prefix_join(prefix, "qrels"),
                            "train.tsv")

    pseudo_qrels = filter.load_pseudo_qrels(
        rel_file, skip_header=True, load_score=False)
    pseudo_qrels = [qrel for qrel in pseudo_qrels if qrel[0] in queries]

    pseudo_qrels_score = filter.score_pseudo_qrels(pseudo_qrels,
                                                   **cross_config)
    score_path = os.path.join(pseudo_data_folder,
                              prefix_join(
                                  target_prefix if target_prefix else prefix, "score"),
                              "train.tsv")
    logging.info(f"Saving score file at {score_path}")
    filter.save_tsv(pseudo_qrels_score, score_path, sort=True)

    return os.path.basename(os.path.dirname(score_path))

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def create_dev(data_folder,
               pseudo_data_folder,
               prefix="gen",
               openai_key=None,
               skip_top_n=50,
               gpt_pos=200,
               nneg=200,
               mine_batch_size=1024,
               device="cuda:0",
               retrievers=["bm25", "msmarco-distilbert-base-v3",
                           "msmarco-MiniLM-L-6-v3"]
               ):

    logging.info("Creating dev set")

    if "dev.tsv" in os.listdir(os.path.join(data_folder, "qrels")):
        logging.info("There's dev set in dataset, mining neg samples")

        miner = NegativeMiner(
            data_folder,
            None,
            split="dev",
            retrievers=retrievers,
            retriever_score_functions=[
                retriever_mapping[retriever] for retriever in retrievers],
            nneg=nneg,
            use_train_qrels=False,
            batch_size=mine_batch_size,
            device=device,
        )

    else:
        logging.info("Dev does not exist, mining pos samples by chatgpt")

        if openai_key:
            logging.info("Using OpenAI API to mine pos samples")

            gpt_labeler = gptLabeler(openai_key)

            corpus = load_corpus(pseudo_data_folder)
            queries = load_queries(pseudo_data_folder, prefix)

            score_folder_name = prefix_join(prefix, "score")
            qrel_scores = CrossFilter.load_pseudo_qrels(rel_file=os.path.join(
                pseudo_data_folder, score_folder_name, "train.tsv"),
                skip_header=True,
                load_score=True)
            qrel_scores = sorted(qrel_scores, key=lambda x: x[2], reverse=True)

            response_path = os.path.join(pseudo_data_folder,
                                         "gpt-response.tsv")
            pos_pairs = []
            saved_response_keys = set()

            if os.path.exists(response_path):
                with open(response_path, 'r') as f:
                    tsv_file = csv.reader(
                        f,
                        delimiter="\t",
                    )
                    for line in tsv_file:
                        saved_response_keys.add(line[0] + line[1])
                        if str(line[2]) == "1":
                            pos_pairs.append((line[0], line[1], 1))

                logging.info(
                    f"Load {len(saved_response_keys)} from {response_path}, {len(pos_pairs)} positive samples"
                )

            with open(response_path, "a") as response_file:

                writer = csv.writer(response_file,
                                    delimiter='\t',
                                    lineterminator='\n')

                for qid, doc_id, score in qrel_scores:

                    if len(pos_pairs) >= gpt_pos:

                        gpt_qrels_path = os.path.join(pseudo_data_folder,
                                                      "gpt-qrels", "dev.tsv")
                        os.makedirs(os.path.dirname(gpt_qrels_path),
                                    exist_ok=True)

                        logging.info(f"pos_pairs examples: {pos_pairs[:3]}")
                        logging.info("Saving gpt qrels to " + gpt_qrels_path)

                        with open(gpt_qrels_path, "w") as f:
                            f.write("query-id\tcorpus-id\tscore\n")
                            for qid, doc_id, _ in pos_pairs:
                                f.write(f"{qid}\t{doc_id}\t1\n")

                        break

                    if str(qid) + str(
                            doc_id) in saved_response_keys or sigmoid(
                                score) > 0.5:
                        continue

                    logging.info(
                        f"Query: {qid}, Doc: {doc_id}, Score: {score}")

                    gpt_pred = gpt_labeler.label(
                        query=queries[qid],
                        passage=corpus[doc_id]["title"] + " " +
                        corpus[doc_id]["text"])

                    if str(gpt_pred) in ["0", "1"]:
                        writer.writerow([qid, doc_id, gpt_pred])

                    logging.info(
                        f"GPT: {gpt_pred} Total pos: {len(pos_pairs)}")

                    if str(gpt_pred) == "1":
                        pos_pairs.append((qid, doc_id, 1))

            miner = NegativeMiner(
                pseudo_data_folder,
                None,
                split="dev",
                corpus_file='corpus.jsonl',
                query_file=prefix_join(prefix, 'queries.jsonl'),
                qrels_folder='gpt-qrels',
                retrievers=retrievers,
                retriever_score_functions=[
                    retriever_mapping[retriever] for retriever in retrievers],
                nneg=nneg,
                use_train_qrels=False,
                batch_size=mine_batch_size,
                device=device,
            )

        else:
            logging.info("OpenAI API key not provided")
            return

    miner.corpus = load_corpus(pseudo_data_folder)
    mine_result = miner.run(
        os.path.join(pseudo_data_folder, "dev-hard-negatives.jsonl"))

    dev_samples = {}
    corpus = load_corpus(data_folder)  # use original corpus

    logging.info(f"Generating dev samples, skip top {skip_top_n} negs")

    for item in mine_result:
        sample = {
            "query": miner.gen_queries[item["qid"]],
            "positive": set(),
            "negative": set(),
        }
        sample["positive"].update(item["pos"])

        for model, neg_doc_ids in item["neg"].items():
            neg_doc_ids = neg_doc_ids[skip_top_n:]
            sample["negative"].update(neg_doc_ids)

        sample["negative"].difference_update(sample["positive"])
        sample["positive"] = list(
            miner._get_doc(doc_id) for doc_id in sample["positive"])
        sample["negative"] = list(
            miner._get_doc(doc_id) for doc_id in sample["negative"])
        dev_samples[item["qid"]] = sample

    dev_samples_path = os.path.join(pseudo_data_folder, "dev_samples.json")
    logging.info(f"Saving dev samples to {dev_samples_path}")

    with open(dev_samples_path, "w") as f:
        json.dump(dev_samples, f)

    return dev_samples_path


def create_negative_pools(
    pseudo_data_folder,
    original_data_folder,
    prefix="gen",
    top_n=100000,
    limit_score=0.5,
    nneg=100,
    batch_size=1024,
    device="cuda:0",
    hard_neg_name="train-cross-hard-negatives.jsonl",
    retrievers=["bm25", "msmarco-distilbert-base-v3",
                "msmarco-MiniLM-L-6-v3"],
    bm25_index=None,
):
    assert (bm25_index is None) or ((bm25_index is not None) and "bm25" in retrievers)

    corpus = load_corpus(original_data_folder)
    queries = load_queries(pseudo_data_folder, prefix)
    filter = CrossFilter(corpus=corpus, queries=queries)

    qrels_score = CrossFilter.load_pseudo_qrels(os.path.join(
        pseudo_data_folder, prefix_join(prefix, "score"), "train.tsv"),
        load_score=True)

    # remove short queries
    qrels_score = filter.remove_short_queries(qrels_score, min_length=5)
    qrels_score = filter.remove_low_score(
        qrels_score, limit_score, sigmoid=True)
    qrels_score = CrossFilter.get_top_n(qrels_score, top_n)
    qrels_score = CrossFilter.set_score(qrels_score, 1)
    
    top_n = len(qrels_score)
    logging.info(f"Saving top {top_n} qrels to {pseudo_data_folder}")
    top_n_qrels = os.path.join(pseudo_data_folder, f"top_{top_n}_qrels",
                               "train.tsv")
    CrossFilter.save_tsv(qrels_score, output_path=top_n_qrels, sort=True)

    miner = NegativeMiner(
        pseudo_data_folder,
        None,
        split="train",
        corpus_file='corpus.jsonl',
        query_file=prefix_join(prefix, 'queries.jsonl'),
        qrels_folder=f"top_{top_n}_qrels",
        retrievers=retrievers,
        retriever_score_functions=[
            retriever_mapping[retriever] for retriever in retrievers],
        nneg=nneg,
        use_train_qrels=False,
        batch_size=batch_size,
        device=device,
        bm25_index=bm25_index,
    )
    miner.corpus = corpus
    miner.run(
        os.path.join(pseudo_data_folder, hard_neg_name))
