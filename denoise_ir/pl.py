from itertools import cycle, islice
from sentence_transformers import CrossEncoder
from gpl.toolkit import HardNegativeDataset
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from tqdm import tqdm
import os
import logging

logger = logging.getLogger(__name__)


def hard_negative_collate_fn(batch):
    query_id, pos_id, neg_id = zip(*[example.guid for example in batch])
    query, pos, neg = zip(*[example.texts for example in batch])
    return (query_id, pos_id, neg_id), (query, pos, neg)


class PseudoLabeler(object):

    def __init__(
        self,
        generated_path,
        gen_queries,
        corpus,
        total_steps,
        batch_size,
        cross_encoder,
        max_seq_length,
        hard_negatives="hard-negatives.jsonl",
        cross_encoder_batch_size=32,
        device="cuda",
    ):

        assert cross_encoder_batch_size % batch_size == 0
        self.cross_encoder_batch_size = cross_encoder_batch_size

        assert hard_negatives in os.listdir(generated_path)
        fpath_hard_negatives = os.path.join(generated_path, hard_negatives)
        self.cross_encoder = CrossEncoder(cross_encoder, device=device)
        hard_negative_dataset = HardNegativeDataset(fpath_hard_negatives,
                                                    gen_queries, corpus)
        self.hard_negative_dataloader = DataLoader(hard_negative_dataset,
                                                   shuffle=True,
                                                   batch_size=batch_size,
                                                   drop_last=True)
        self.hard_negative_dataloader.collate_fn = hard_negative_collate_fn
        self.output_path = os.path.join(generated_path,
                                        "gpl-training-data.tsv")
        self.total_steps = total_steps

        # retokenization
        self.retokenizer = AutoTokenizer.from_pretrained(cross_encoder)
        self.max_seq_length = max_seq_length

    def retokenize(self, texts):
        # We did this retokenization for two reasons:
        # (1) Setting the max_seq_length;
        # (2) We cannot simply use CrossEncoder(cross_encoder, max_length=max_seq_length),
        # since the max_seq_length will then be reflected on the concatenated sequence,
        # rather than the two sequences independently
        texts = list(map(lambda text: text.strip(), texts))
        features = self.retokenizer(
            texts,
            padding=True,
            truncation="longest_first",
            return_tensors="pt",
            max_length=self.max_seq_length,
        )
        decoded = self.retokenizer.batch_decode(
            features["input_ids"],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        return decoded

    def create_batch(self, query_id, pos_id, neg_id, query, pos, neg):

        scores = self.cross_encoder.predict(
            list(zip(query, pos)) + list(zip(query, neg)),
            batch_size=self.cross_encoder_batch_size,
            show_progress_bar=False)

        labels = scores[:len(query)] - scores[len(query):]
        labels = (labels.tolist()
                  )  # Using `tolist` will keep more precision digits!!!

        batch_gpl = map(
            lambda quad: "\t".join((*quad[:3], str(quad[3]))) + "\n",
            zip(query_id, pos_id, neg_id, labels),
        )
        return batch_gpl

    def run(self):
        num_data_points = len(self.hard_negative_dataloader.dataset)
        batch_size = self.hard_negative_dataloader.batch_size

        if num_data_points < batch_size:
            raise ValueError(
                f"Batch size larger than number of data points / generated queries "
                f"(batch size: {batch_size}, "
                f"number of data points / generated queries: {num_data_points})"
            )

        # header: 'query_id', 'positive_id', 'negative_id', 'pseudo_label_margin'
        data = []
        hard_negative_iterator = cycle(iter(self.hard_negative_dataloader))
        logger.info("Begin pseudo labeling")

        cross_interval = self.cross_encoder_batch_size // batch_size
        batch_data = {
            "query_id": [],
            "pos_id": [],
            "neg_id": [],
            "query": [],
            "pos": [],
            "neg": [],
        }

        for i in tqdm(range(1, self.total_steps + 1)):
            try:
                batch = next(hard_negative_iterator)
            except StopIteration:
                hard_negative_iterator = iter(self.hard_negative_dataloader)
                batch = next(hard_negative_iterator)

            (query_id, pos_id, neg_id), (query, pos, neg) = batch
            query, pos, neg = [
                self.retokenize(texts) for texts in [query, pos, neg]
            ]

            batch_data["query_id"].extend(query_id)
            batch_data["pos_id"].extend(pos_id)
            batch_data["neg_id"].extend(neg_id)
            batch_data["query"].extend(query)
            batch_data["pos"].extend(pos)
            batch_data["neg"].extend(neg)

            if i % cross_interval == 0:
                data.extend(self.create_batch(**batch_data))

                for k in batch_data.keys():
                    batch_data[k].clear()

        if batch_data["query_id"]:
            data.extend(self.create_batch(**batch_data))

        logger.info("Done pseudo labeling and saving data")
        with open(self.output_path, "w") as f:
            f.writelines(data)

        logger.info(f"Saved GPL-training data to {self.output_path}")
