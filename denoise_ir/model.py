# %%
from sentence_transformers import CrossEncoder
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
from typing import Dict, Type, Callable, List
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm, trange
import os
import torch
from sentence_transformers.evaluation import SentenceEvaluator
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import logging
import numpy as np
import random

logger = logging.getLogger(__name__)
# %%


class DenoiseCrossEncoder(CrossEncoder):
    """ Denoise version of CrossEncoder
        https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/cross_encoder/CrossEncoder.py
    """

    def __init__(self, model_names: list, num_labels: int = None, max_length: int = None, device: str = None, tokenizer_args: Dict = {},
                 automodel_args: Dict = {}, default_activation_function=None, test_evaluator=None):
        super().__init__(
            model_name=model_names[0], num_labels=num_labels, max_length=max_length, device=device, tokenizer_args=tokenizer_args,
            automodel_args=automodel_args, default_activation_function=default_activation_function
        )
        self.model_names = model_names  # ["0_modelxxx", "1_modelxxx", ...]
        self.model_tokenizer = {
            model_name: AutoTokenizer.from_pretrained(model_name, **tokenizer_args) for model_name in model_names
        }
        self.n_models = len(model_names)

        for i, m in enumerate(model_names):
            logger.info(f"{i}: {m}")

        self.model_list = nn.ModuleList()
        self.model_list.append(self.model)

        for model in self.model_names[1:]:  # append more models
            self.model_list.append(
                AutoModelForSequenceClassification.from_pretrained(
                    model, config=AutoConfig.from_pretrained(model), **automodel_args)
            )
        self.test_evaluator = test_evaluator

    def denoise_batching_collate(self, batch):

        texts = [[] for _ in range(len(batch[0].texts))]
        labels = []

        for example in batch:
            for idx, text in enumerate(example.texts):
                texts[idx].append(text.strip())
            labels.append(example.label)

        labels = torch.tensor(labels, dtype=torch.float if self.config.num_labels ==
                              1 else torch.long).to(self._target_device)
        model_tokenized = {}

        for model_name, tokenizer in self.model_tokenizer.items():
            tokenized = tokenizer(*texts, padding=True, truncation='longest_first',
                                  return_tensors="pt", max_length=self.max_length)

            for name in tokenized:
                tokenized[name] = tokenized[name].to(self._target_device)

            model_tokenized[model_name] = tokenized

        return model_tokenized, labels

    def fit(self,
            train_dataloader: DataLoader,
            gamma: float,
            denoise_warmup_steps: int = 0,
            random_batch_warmup_steps: int = 0,
            random_batch_warmup_p: float = 0.2,
            evaluator: SentenceEvaluator = None,
            epochs: int = 1,
            loss_fct=None,
            activation_fct=nn.Identity(),
            scheduler: str = 'WarmupLinear',
            warmup_steps: int = 10000,
            optimizer_class: Type[Optimizer] = torch.optim.AdamW,
            optimizer_params: Dict[str, object] = {'lr': 2e-5},
            weight_decay: float = 0.01,
            evaluation_steps: int = 0,
            output_path: str = None,
            save_best_model: bool = True,
            max_grad_norm: float = 1,
            use_amp: bool = False,
            callback: Callable[[float, int, int], None] = None,
            show_progress_bar: bool = True
            ):
        """
        Train the model with the given training objective
        Each training objective is sampled in turn for one batch.
        We sample only as many batches from each objective as there are in the smallest one
        to make sure of equal training with each dataset.

        :param train_dataloader: DataLoader with training InputExamples
        :param evaluator: An evaluator (sentence_transformers.evaluation) evaluates the model performance during training on held-out dev data. It is used to determine the best model that is saved to disc.
        :param epochs: Number of epochs for training
        :param loss_fct: Which loss function to use for training. If None, will use nn.BCEWithLogitsLoss() if self.config.num_labels == 1 else nn.CrossEntropyLoss()
        :param activation_fct: Activation function applied on top of logits output of model.
        :param scheduler: Learning rate scheduler. Available schedulers: constantlr, warmupconstant, warmuplinear, warmupcosine, warmupcosinewithhardrestarts
        :param warmup_steps: Behavior depends on the scheduler. For WarmupLinear (default), the learning rate is increased from o up to the maximal learning rate. After these many training steps, the learning rate is decreased linearly back to zero.
        :param optimizer_class: Optimizer
        :param optimizer_params: Optimizer parameters
        :param weight_decay: Weight decay for model parameters
        :param evaluation_steps: If > 0, evaluate the model using evaluator after each number of training steps
        :param output_path: Storage path for the model and evaluation files
        :param save_best_model: If true, the best model (according to evaluator) is stored at output_path
        :param max_grad_norm: Used for gradient normalization.
        :param use_amp: Use Automatic Mixed Precision (AMP). Only for Pytorch >= 1.6.0
        :param callback: Callback function that is invoked after each evaluation.
                It must accept the following three parameters in this order:
                `score`, `epoch`, `steps`
        :param show_progress_bar: If True, output a tqdm progress bar
        """

        # logger
        os.makedirs(os.path.join(output_path, 'logs'), exist_ok=True)
        self.writer = SummaryWriter(log_dir=os.path.join(output_path, 'logs'))

        train_dataloader.collate_fn = self.denoise_batching_collate

        if use_amp:
            from torch.cuda.amp import autocast
            scaler = torch.cuda.amp.GradScaler()

        self.model_list.to(self._target_device)

        if output_path is not None:
            os.makedirs(output_path, exist_ok=True)

        self.best_score = -9999999
        num_train_steps = int(len(train_dataloader) * epochs)
        logger.info("Total steps: {}".format(num_train_steps))
        # logger.info("***** Running training *****")
        logger.info("***** Running training *****")
        # Prepare optimizers
        param_optimizer = list(self.model_list.named_parameters())

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(
                nd in n for nd in no_decay)], 'weight_decay': weight_decay},
            {'params': [p for n, p in param_optimizer if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        optimizer = optimizer_class(
            optimizer_grouped_parameters, **optimizer_params)

        if isinstance(scheduler, str):
            scheduler = SentenceTransformer._get_scheduler(
                optimizer, scheduler=scheduler, warmup_steps=warmup_steps, t_total=num_train_steps)

        if loss_fct is None:
            loss_fct = nn.BCEWithLogitsLoss(
            ) if self.config.num_labels == 1 else nn.CrossEntropyLoss()

        # the activation function applied on top of the logits output of the model
        output_activate_fct = nn.Softmax(
            dim=-1) if self.config.num_labels > 1 else nn.Sigmoid()

        agg_loss_fct = nn.KLDivLoss(
            reduction='batchmean') if self.config.num_labels > 1 else nn.BCEWithLogitsLoss(reduction='mean')

        skip_scheduler = False
        training_steps = 0

        self.best_test_matrix = {}

        logger.info(f"Random batch warmup steps: {random_batch_warmup_steps}")

        assert 0<=random_batch_warmup_p<1, "random_batch_warmup_p must be in [0,1)"
        for epoch in trange(epochs, desc="Epoch", disable=not show_progress_bar, ncols=100):

            self.model_list.zero_grad()
            self.model_list.train()

            for features, labels in tqdm(train_dataloader, desc="Iteration", smoothing=0.05, disable=not show_progress_bar):
                """
                features: {
                    "model1": tokenized features for model1,
                    "model2": tokenized features for model2,
                }
                """
                if use_amp:
                    with autocast():
                        loss_value = 0
                        label_loss = 0
                        predictions = []

                        for i in range(self.n_models):

                            if training_steps < random_batch_warmup_steps:
                                p = random.random()
                                if p < random_batch_warmup_p:
                                    continue

                            # the prediction before the softmax/sigmoid layer
                            # (batch_size, num_labels)
                            model_prediction = self.model_list[i](
                                **features[self.model_names[i]], return_dict=True
                            )
                            logits = activation_fct(model_prediction.logits)

                            if self.config.num_labels == 1:
                                logits = logits.view(-1)  # batch_size

                            loss = loss_fct(logits, labels)
                            label_loss += loss
                            self.writer.add_scalar(
                                f"Loss/{i}-{self.model_names[i]}", loss.item(), training_steps)
                            predictions.append(logits)

                        if len(predictions) == 0:
                            continue

                        label_loss /= len(predictions)

                        self.writer.add_scalar(
                            'Loss/label_loss', label_loss, training_steps)

                        loss_value += label_loss

                        # calculate aggregation loss
                        if training_steps >= denoise_warmup_steps \
                                and self.n_models > 1 and gamma > 0.0  \
                                and len(predictions) > 1:

                            agg_loss = 0.0
                            # list of (batch_size, num_labels) if n_models > 1
                            # else (batch_size)
                            pred_probs = [output_activate_fct(
                                p) for p in predictions]
                            agg_prob = torch.stack(
                                pred_probs, dim=0).mean(0).detach().clone()

                            for pred in predictions:
                                agg_loss += agg_loss_fct(pred, agg_prob)

                            agg_loss /= len(predictions)
                            self.writer.add_scalar(
                                'Loss/agg_loss', agg_loss, training_steps)
                            loss_value += gamma * agg_loss

                    self.writer.add_scalar(
                        'Loss/loss_value', loss_value, training_steps)
                    scale_before_step = scaler.get_scale()
                    scaler.scale(loss_value).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model_list.parameters(), max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    skip_scheduler = scaler.get_scale() != scale_before_step
                else:
                    loss_value = 0
                    label_loss = 0
                    predictions = []
                    for i in range(self.n_models):
                        # the prediction before the softmax layer or sigmoid layer
                        # (batch_size, num_labels)
                        model_prediction = self.model_list[i](
                            **features[self.model_names[i]], return_dict=True
                        )

                        logits = activation_fct(model_prediction.logits)

                        if self.config.num_labels == 1:
                            logits = logits.view(-1)  # batch_size
                        label_loss += loss_fct(logits, labels)
                        self.writer.add_scalar(
                            'Loss/label_loss', label_loss, training_steps)
                        # list of (batch_size, num_labels) if n_models > 1
                        # else (batch_size)
                        predictions.append(logits)
                    loss_value += label_loss / self.n_models

                    # calculate aggregation loss
                    if self.n_models > 1 and gamma > 0.0:
                        agg_loss = 0.0
                        # list of (batch_size, num_labels) if n_models > 1
                        # else (batch_size)
                        pred_probs = [output_activate_fct(
                            p) for p in predictions]
                        agg_prob = torch.stack(
                            pred_probs, dim=0).mean(0)

                        for pred in predictions:
                            agg_loss += agg_loss_fct(pred, agg_prob)

                        agg_loss /= self.n_models
                        self.writer.add_scalar(
                            'Loss/agg_loss', agg_loss, training_steps)
                        loss_value += gamma * agg_loss
                    self.writer.add_scalar(
                        'Loss/loss_value', loss_value, training_steps)
                    loss_value.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_grad_norm)
                    optimizer.step()

                optimizer.zero_grad()

                if not skip_scheduler:
                    scheduler.step()

                training_steps += 1

                if evaluation_steps > 0 and training_steps % evaluation_steps == 0:

                    if evaluator is not None:
                        self._eval_during_training(
                            evaluator, output_path, save_best_model, epoch, training_steps, callback)

                    if self.test_evaluator is not None:
                        self.test_during_training(training_steps, output_path)

                    self.model_list.zero_grad()
                    self.model_list.train()

            if evaluator is not None:
                self._eval_during_training(
                    evaluator, output_path, save_best_model, epoch, -1, callback)

    def _eval_during_training(self, evaluator, output_path, save_best_model, epoch, steps, callback):
        """Runs evaluation during the training"""
        if evaluator is not None:
            score = evaluator(self, output_path=output_path,
                              epoch=epoch, steps=steps)

            self.writer.add_scalar(
                'eval/final' if steps == -1 else 'eval/training', score, steps)
            if callback is not None:
                callback(score, epoch, steps)
            if score > self.best_score:
                self.best_score = score
                if save_best_model:
                    self.save(output_path)

    def test_during_training(self, training_steps: int, output_path=None):
        """Runs test during the training"""
        logging.info("Test the model on test dataset...")

        all_models_best = {}

        for i, model_name in enumerate(self.model_names):
            model = CrossEncoder(
                model_name,
                num_labels=self.config.num_labels,
                max_length=self.max_length,
                device=self._target_device,
            )
            model.model.load_state_dict(self.model_list[i].state_dict())
            test_report = self.test_evaluator(model)

            for report in test_report:
                """
                "test_matrix": f"Top@{k} - {self.test_matrix}",
                "score": result[self.test_matrix]
                """
                matrix, score = report["test_matrix"], report["score"]
                all_models_best[matrix] = max(
                    score, all_models_best.get(matrix, -1))

                logging.info(
                    f"Idx {i} - {model_name}")
                logging.info(
                    f" Test {matrix} at steps {training_steps} is {score:.4f}")

                self.writer.add_scalar(
                    f"test/{i}-{model_name}-{matrix}", score, training_steps)

                if score > self.best_test_matrix.get(matrix, -1):
                    self.best_test_matrix[matrix] = score
                    if output_path is not None:
                        op = os.path.join(
                            output_path, f"best_test_{matrix.replace(' ', '_').replace('/', '_')}")
                        logging.info(
                            f"Save best test model {i}-{model_name} to {op}...")
                        model.save(op)

        for matrix, score in all_models_best.items():
            self.writer.add_scalar(f"test/{matrix}", score, training_steps)

    def parallel_predict(self, sentences: List[List[str]],
                         batch_size: int = 32,
                         show_progress_bar: bool = None,
                         num_workers: int = 0,
                         activation_fct=None,
                         apply_softmax=False,
                         convert_to_numpy: bool = True,
                         convert_to_tensor: bool = False,
                         device_ids=[0]
                         ):
        """
        Performs predicts with the CrossEncoder on the given sentence pairs.

        :param sentences: A list of sentence pairs [[Sent1, Sent2], [Sent3, Sent4]]
        :param batch_size: Batch size for encoding
        :param show_progress_bar: Output progress bar
        :param num_workers: Number of workers for tokenization
        :param activation_fct: Activation function applied on the logits output of the CrossEncoder. If None, nn.Sigmoid() will be used if num_labels=1, else nn.Identity
        :param convert_to_numpy: Convert the output to a numpy matrix.
        :param apply_softmax: If there are more than 2 dimensions and apply_softmax=True, applies softmax on the logits output
        :param convert_to_tensor:  Conver the output to a tensor.
        :param device_ids: List of GPU device ids to use.
        :return: Predictions for the passed sentence pairs
        """
        input_was_string = False
        # Cast an individual sentence to a list with length 1
        if isinstance(sentences[0], str):
            sentences = [sentences]
            input_was_string = True

        self._target_device = f"cuda:{device_ids[0]}"

        inp_dataloader = DataLoader(sentences, batch_size=batch_size,
                                    collate_fn=self.smart_batching_collate_text_only, num_workers=num_workers, shuffle=False)

        if show_progress_bar is None:
            show_progress_bar = (logger.getEffectiveLevel(
            ) == logging.INFO or logger.getEffectiveLevel() == logging.DEBUG)

        iterator = inp_dataloader
        if show_progress_bar:
            iterator = tqdm(inp_dataloader, desc="Batches")

        if activation_fct is None:
            activation_fct = self.default_activation_function

        pred_scores = []

        model = torch.nn.DataParallel(
            self.model, device_ids=device_ids)
        model.to(self._target_device)
        model.eval()

        with torch.no_grad():
            for features in iterator:
                model_predictions = model(**features, return_dict=True)
                logits = activation_fct(model_predictions.logits)

                if apply_softmax and len(logits[0]) > 1:
                    logits = torch.nn.functional.softmax(logits, dim=1)
                pred_scores.extend(logits)

        if self.config.num_labels == 1:
            pred_scores = [score[0] for score in pred_scores]

        if convert_to_tensor:
            pred_scores = torch.stack(pred_scores)
        elif convert_to_numpy:
            pred_scores = np.asarray(
                [score.cpu().detach().numpy() for score in pred_scores])

        if input_was_string:
            pred_scores = pred_scores[0]

        return pred_scores
