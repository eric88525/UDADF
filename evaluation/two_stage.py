import logging
import pathlib, os
import sys
import pickle
sys.path.append(os.getcwd)
from denoise.utils import load_corpus
import argparse
from time import time

    
def parse_args():
    
    parser = argparse.ArgumentParser(description="Argument Parser")
    
    parser.add_argument('--rerank_model', type=str)
    parser.add_argument('--retrieval_model', type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--ori_dataset', type=str, help='the path to original dataset')
    parser.add_argument('--retrieval_batch_size', type=int, default=512)
    parser.add_argument('--rerank_batch_size', type=int, default=1300)
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--top_k', type=int, nargs='+', default=[100])
    parser.add_argument('--saved_retrieval_result', type=str, default=None)
    args = parser.parse_args()
    return args

def main(args):
     
    from beir import util
    from beir.retrieval import models
    from beir.datasets.data_loader import GenericDataLoader
    from beir.retrieval.evaluation import EvaluateRetrieval
    from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
    from beir.reranking.models import CrossEncoder
    from beir.reranking import Rerank  
    
    os.makedirs(f"./two_stage_logs/{args.dataset}", exist_ok=True)
    assert args.retrieval_model.find(args.dataset) != -1

    logging.info("=======================================")
    logging.info(f"Retrieval model: {args.retrieval_model}")
    logging.info(f"Rerank model: {args.rerank_model}")
    logging.info("=======================================")

    #### Download nfcorpus.zip dataset and unzip the dataset
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(args.dataset)
    out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
    data_path = util.download_and_unzip(url, out_dir)

    #### Provide the data path where nfcorpus has been downloaded and unzipped to the data loader
    # data folder would contain these files:
    # (1) nfcorpus/corpus.jsonl  (format: jsonlines)
    # (2) nfcorpus/queries.jsonl (format: jsonlines)
    # (3) nfcorpus/qrels/test.tsv (format: tsv ("\t"))

    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")

    #### Dense Retrieval using SBERT (Sentence-BERT) ####
    #### Provide any pretrained sentence-transformers model
    #### The model was fine-tuned using cosine-similarity.
    #### Complete list - https://www.sbert.net/docs/pretrained_models.html

    # ================== Retrieval ==================
    corpus = load_corpus(args.ori_dataset, True)
    results = None

    if os.path.exists(args.saved_retrieval_result):
        y = input('use temp file? (y/n)')
        if y == 'y' and args.saved_retrieval_result.endswith('.pkl'):
            with open(args.saved_retrieval_result, "rb") as f:
                results = pickle.load(f)
    else:
        print('temp file not exist')
        os.makedirs('retrieval_temps', exist_ok=True)

    if results is  None:
        model = DRES(models.SentenceBERT(args.retrieval_model), batch_size=args.retrieval_batch_size, corpus_chunk_size=512*9999)
        retriever = EvaluateRetrieval(model, score_function="dot")

        #### Retrieve dense results (format of results is identical to qrels)
        start_time = time()
        results = retriever.retrieve(corpus, queries)
        end_time = time()

        print("Time taken to retrieve: {:.2f} seconds".format(end_time - start_time))
        retriever.evaluate(qrels, results, retriever.k_values)

        with open(f'retrieval_temps/{args.dataset}_{args.split}.pkl', 'wb') as f:
            pickle.dump(results, f)

    # ================== Reranking ==================
    if os.path.exists(args.rerank_model):
        model = CrossEncoder(model_path=args.rerank_model, max_length=300, device="cuda")
        reranker = Rerank(model, batch_size=args.rerank_batch_size)
        for k in args.top_k:

            rerank_results = reranker.rerank(
                corpus=corpus,
                queries=queries,
                results=results,
                top_k=k
            )
            EvaluateRetrieval.evaluate(qrels, rerank_results, [1, 3, 5, 10, 100, 1000])
    else:
        print('rerank model not exist')
    
if __name__ == "__main__":
    args = parse_args()

    #### Just some code to print debug information to stdout
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        filename=f"./two_stage_logs/{args.dataset}/log_{args.split}.txt",
                        )
    main(args)