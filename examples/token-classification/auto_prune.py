# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Fine-tuning the library models for named entity recognition on CoNLL-2003 (Bert or Roberta). """

import logging
import os
import sys
from datetime import datetime

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import torch

import numpy as np
from seqeval.metrics import f1_score, precision_score, recall_score
from torch import nn
from torch.utils.data import Subset, RandomSampler, DataLoader, DistributedSampler
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    EvalPrediction,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from utils_ner import NerDataset, Split, get_labels, align_predictions, compute_metrics
from src.transformers import DefaultDataCollator
from head_score_predictor import MLPPredictor, SparseLoss, HardConcretePredictor

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    use_fast: bool = field(default=False, metadata={"help": "Set this flag to use fast tokenization."})
    # If you want to tweak more attributes on your tokenizer, you should do it in a distinct script,
    # or just modify its tokenizer_config.json.
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    data_dir: str = field(
        metadata={"help": "The input data dir. Should contain the .txt files for a CoNLL-2003-formatted task."}
    )
    labels: Optional[str] = field(
        metadata={"help": "Path to a file containing all labels. If not specified, CoNLL-2003 labels are used."}
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )

    predictor : str = field(
        default='mlp', metadata={"help": "predictor type"}
    )

    metric_name: str = field(
        default='f1', metadata={"help": "metric name for evalute pruned model"}
    )

    sparse_ratio: float = field(
        default=0.1, metadata={'help': "sparse loss ratio"}
    )
    predictor_lr: float = field(
        default=1e-3, metadata={'help': "head predictor learning rate"}
    )

    epoch_num: int = field(
        default=10, metadata={'help': "epoch of training head predictor"}
    )


def evaluate_masked_model(args, model, eval_dataloader, head_mask):
    preds, labels = None, None
    for step, inputs in enumerate(tqdm(eval_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])):
        for k, v in inputs.items():
            inputs[k] = v.to(args.device)
            # logger.info(k)

        # Do a forward pass (not with torch.no_grad() since we need gradients for importance score - see below)
        model.eval()
        with torch.no_grad():
            outputs = model(**inputs, head_mask=head_mask)
            loss, logits, all_attentions = (
                outputs[0],
                outputs[1],
                outputs[-1],
            )  # Loss and logits are the first, attention the last

        # Also store our logits/labels if we want to compute metrics afterwards
        if preds is None:
            preds = logits.detach().cpu().numpy()
            labels = inputs["labels"].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            labels = np.append(labels, inputs["labels"].detach().cpu().numpy(), axis=0)
    return preds, labels


def entropy(p):
    """ Compute the entropy of a probability distribution """
    plogp = p * torch.log(p)
    plogp[p == 0] = 0
    return -plogp.sum(dim=-1)


def print_2d_tensor(tensor):
    """ Print a 2D tensor """
    logger.info("lv, h >\t" + "\t".join(f"{x + 1}" for x in range(len(tensor))))
    for row in range(len(tensor)):
        if tensor.dtype != torch.long:
            logger.info(f"layer {row + 1}:\t" + "\t".join(f"{x:.5f}" for x in tensor[row].cpu().data))
        else:
            logger.info(f"layer {row + 1}:\t" + "\t".join(f"{x:d}" for x in tensor[row].cpu().data))


def compute_heads_importance(
        args, model, eval_dataloader, compute_entropy=True, compute_importance=True, head_mask=None,
        actually_pruned=False, metric_name='f1'
):
    """ This method shows how to compute:
        - head attention entropy
        - head importance scores according to http://arxiv.org/abs/1905.10650
    """
    # Prepare our tensors
    n_layers, n_heads = model.config.num_hidden_layers, model.config.num_attention_heads
    head_importance = torch.zeros(n_layers, n_heads).to(args.device)
    attn_entropy = torch.zeros(n_layers, n_heads).to(args.device)

    if head_mask is None:
        head_mask = torch.ones(n_layers, n_heads).to(args.device)

    head_mask.requires_grad_(requires_grad=True)
    # If actually pruned attention multi-head, set head mask to None to avoid shape mismatch
    if actually_pruned:
        head_mask = None

    preds = None
    labels = None
    tot_tokens = 0.0
    for step, inputs in enumerate(tqdm(eval_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])):
        for k, v in inputs.items():
            inputs[k] = v.to(args.device)
            # logger.info(k)

        # Do a forward pass (not with torch.no_grad() since we need gradients for importance score - see below)
        outputs = model(**inputs, head_mask=head_mask)
        loss, logits, all_attentions = (
            outputs[0],
            outputs[1],
            outputs[-1],
        )  # Loss and logits are the first, attention the last
        loss.backward()  # Backpropagate to populate the gradients in the head mask
        # logger.info('loss id: %d' % id(loss))
        # del loss
        if compute_entropy:
            for layer, attn in enumerate(all_attentions):
                masked_entropy = entropy(attn.detach()) * inputs["attention_mask"].float().unsqueeze(1)
                attn_entropy[layer] += masked_entropy.sum(-1).sum(0).detach()

        if compute_importance:
            head_importance += head_mask.grad.abs().detach()
        # Also store our logits/labels if we want to compute metrics afterwards
        if preds is None:
            preds = logits.detach().cpu().numpy()
            labels = inputs["labels"].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            labels = np.append(labels, inputs["labels"].detach().cpu().numpy(), axis=0)

        tot_tokens += inputs["attention_mask"].float().detach().sum().data

    # Normalize
    attn_entropy /= tot_tokens
    head_importance /= tot_tokens
    # Layerwise importance normalization
    if not False:  # args.dont_normalize_importance_by_layer:
        exponent = 2
        norm_by_layer = torch.pow(torch.pow(head_importance, exponent).sum(-1), 1 / exponent)
        head_importance /= norm_by_layer.unsqueeze(-1) + 1e-20

    if not False:  # args.dont_normalize_global_importance:
        head_importance = (head_importance - head_importance.min()) / (head_importance.max() - head_importance.min())

    # Print/save matrices
    if compute_entropy:
        np.save(os.path.join(args.output_dir, "attn_entropy.npy"), attn_entropy.detach().cpu().numpy())
    if compute_importance:
        np.save(os.path.join(args.output_dir, "head_importance.npy"), head_importance.detach().cpu().numpy())

    logger.info("Attention entropies")
    print_2d_tensor(attn_entropy)
    logger.info("Head importance scores")
    print_2d_tensor(head_importance)
    logger.info("Head ranked by importance scores")
    head_ranks = torch.zeros(head_importance.numel(), dtype=torch.long, device=args.device)
    head_ranks[head_importance.view(-1).sort(descending=True)[1]] = torch.arange(
        head_importance.numel(), device=args.device
    )
    head_ranks = head_ranks.view_as(head_importance)
    print_2d_tensor(head_ranks)

    return attn_entropy, head_importance, preds, labels


def mask_heads(args, model, eval_dataloader, metric_name='f1', head_num=144, per_iter_mask=1, label_map=None):
    """ This method shows how to mask head (set some heads to zero), to test the effect on the network,
        based on the head importance scores, as described in Michel et al. (http://arxiv.org/abs/1905.10650)
    """
    _, head_importance, preds, labels = compute_heads_importance(args, model, eval_dataloader, compute_entropy=False)
    original_score = compute_metrics(EvalPrediction(preds, labels), label_map)[metric_name]
    logger.info("Pruning: original score: %f, target left head num: %d", original_score, head_num)

    new_head_mask = torch.ones_like(head_importance)
    num_to_mask = per_iter_mask  # max(1, int(new_head_mask.numel() * args.masking_amount))

    while int(new_head_mask.sum()) > head_num:  # current_score >= original_score * args.masking_threshold:
        head_mask = new_head_mask.clone()  # save current head mask
        # heads from least important to most - keep only not-masked heads
        head_importance[head_mask == 0.0] = float("Inf")
        current_heads_to_mask = head_importance.view(-1).sort()[1]
        if len(current_heads_to_mask) <= num_to_mask:
            break

        # mask heads
        current_heads_to_mask = current_heads_to_mask[:num_to_mask]
        logger.info("Heads to mask: %s", str(current_heads_to_mask.tolist()))
        new_head_mask = new_head_mask.view(-1)
        new_head_mask[current_heads_to_mask] = 0.0
        new_head_mask = new_head_mask.view_as(head_mask)
        new_head_mask = new_head_mask.clone().detach()
        print_2d_tensor(new_head_mask)

        # Compute metric and head importance again
        _, head_importance, preds, labels = compute_heads_importance(
            args, model, eval_dataloader, compute_entropy=False, head_mask=new_head_mask, metric_name=metric_name
        )
        # preds = np.argmax(preds, axis=1) if args.output_mode == "classification" else np.squeeze(preds)
        # current_score = glue_compute_metrics(args.task_name, preds, labels)[args.metric_name]
        current_score = compute_metrics(EvalPrediction(preds, labels), label_map)[metric_name]
        logger.info(
            "Masking: current score: %f, remaining heads %d (%.1f percents)",
            current_score,
            new_head_mask.sum(),
            new_head_mask.sum() / new_head_mask.numel() * 100,
        )
    logger.info("Final head mask")
    print_2d_tensor(head_mask)
    np.save(os.path.join(args.output_dir, "head_mask.npy"), head_mask.detach().cpu().numpy())

    return head_mask


def prune_heads(args, model, eval_dataloader, head_mask, metric_name='f1', label_map=None):
    """ This method shows how to prune head (remove heads weights) based on
        the head importance scores as described in Michel et al. (http://arxiv.org/abs/1905.10650)
    """
    # Try pruning and test time speedup
    # Pruning is like masking but we actually remove the masked weights
    before_time = datetime.now()
    _, _, preds, labels = compute_heads_importance(
        args, model, eval_dataloader, compute_entropy=False, compute_importance=False, head_mask=head_mask
    )
    # preds = np.argmax(preds, axis=1) if args.output_mode == "classification" else np.squeeze(preds)
    # score_masking = glue_compute_metrics(args.task_name, preds, labels)[args.metric_name]
    score_masking = compute_metrics(EvalPrediction(predictions=preds, labels=labels, label_map=label_map))[metric_name]
    original_time = datetime.now() - before_time

    original_num_params = sum(p.numel() for p in model.parameters())

    heads_to_prune = dict(
        (layer, (1 - head_mask[layer].long()).nonzero().squeeze().tolist()) for layer in range(len(head_mask))
    )

    assert sum(len(h) for h in heads_to_prune.values()) == (1 - head_mask.long()).sum().item()
    model.prune_heads(heads_to_prune)
    pruned_num_params = sum(p.numel() for p in model.parameters())

    before_time = datetime.now()
    _, _, preds, labels = compute_heads_importance(
        args,
        model,
        eval_dataloader,
        compute_entropy=False,
        compute_importance=False,
        head_mask=None,
        actually_pruned=True,
    )
    # preds = np.argmax(preds, axis=1) if args.output_mode == "classification" else np.squeeze(preds)
    # score_pruning = glue_compute_metrics(args.task_name, preds, labels)[args.metric_name]
    score_pruning = compute_metrics(EvalPrediction(preds, labels), label_map)[metric_name]

    new_time = datetime.now() - before_time

    logger.info(
        "Pruning: original num of params: %.2e, after pruning %.2e (%.1f percents)",
        original_num_params,
        pruned_num_params,
        pruned_num_params / original_num_params * 100,
    )
    logger.info("Pruning: score with masking: %f score with pruning: %f", score_masking, score_pruning)
    logger.info("Pruning: speed ratio (new timing / original timing): %f percents", original_time / new_time * 100)


def search_optimal_heads(args, model, predictor, optimizer, sparse_loss, eval_dataloader, head_score,
                         metric_name='f1', label_map=None, sparse_ratio=0.1):
    _, head_importance, preds, labels = compute_heads_importance(args, model, eval_dataloader,
                                                                 compute_entropy=False,
                                                                 head_mask=head_score,
                                                                 metric_name=metric_name)
    original_score = compute_metrics(EvalPrediction(preds, labels), label_map)[metric_name]
    logger.info("Pruning: current  score: %f",
                original_score)  # , threshold: %f , original_score * args.masking_threshold)
    head_score = predictor(head_importance.transpose(1, 0))

    for step, inputs in enumerate(tqdm(eval_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])):
        for k, v in inputs.items():
            inputs[k] = v.to(args.device)
            # logger.info(k)
        # head_score.requires_grad_(True)
        # Do a forward pass (not with torch.no_grad() since we need gradients for updating predictor
        outputs = model(**inputs, head_mask=head_score)
        loss, logits, all_attentions = (
            outputs[0],
            outputs[1],
            outputs[-1],
        )  # Loss and logits are the first, attention the last
        s_loss = sparse_loss(head_score)
        total = loss + sparse_ratio * s_loss  # 观察到一个现象 经常是同一个head 不同 Layer 都会置成0  不过，效果还是有一定的提升的
        total.backward()  # Backpropagate to populate the gradients in the head mask
        optimizer.step()  # update predictor score

        # compute head_importance again
        no_mask = torch.ones_like(head_score)
        no_mask.requires_grad_(True)
        importance_loss = model(**inputs, head_mask=no_mask)[0]
        head_importance = torch.autograd.grad(importance_loss, no_mask)[0].abs().detach()
        head_score = predictor(head_importance.transpose(1, 0))  # update head score

    logger.info('current total loss: %f' % total.item())
    logger.info('current head score')
    print_2d_tensor(head_score)
    return head_score


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    # Prepare CONLL-2003 task
    labels = get_labels(data_args.labels)
    label_map: Dict[int, str] = {i: label for i, label in enumerate(labels)}
    num_labels = len(labels)
    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        id2label=label_map,
        label2id={label: i for i, label in enumerate(labels)},
        cache_dir=model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast,
    )
    model = AutoModelForTokenClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
    )

    # Distributed and parallel training
    model.to(training_args.device)
    if training_args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[training_args.local_rank],
            output_device=training_args.local_rank, find_unused_parameters=True
        )
    elif training_args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Print/save training arguments
    os.makedirs(training_args.output_dir, exist_ok=True)
    torch.save(training_args, os.path.join(training_args.output_dir, "run_args.bin"))
    logger.info("Training/evaluation parameters %s", training_args)

    eval_dataset = NerDataset(
        data_dir=data_args.data_dir,
        tokenizer=tokenizer,
        labels=labels,
        model_type=config.model_type,
        max_seq_length=data_args.max_seq_length,
        overwrite_cache=data_args.overwrite_cache,
        mode=Split.dev,
    )
    eval_dataset.set_mode('half')
    eval_dataset.set_index(0)  # use first half
    eval_sampler = RandomSampler(eval_dataset) if training_args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=training_args.train_batch_size,
        collate_fn=DefaultDataCollator().collate_batch
    )

    # head_score_predictor = RNNHeadPredictor(12, 12, 128, rnn_layers=2)
    # head_score_predictor = MLPPredictor(12, 12, 128)  # bsz can be important ?
    if data_args.predictor == 'mlp':
        head_score_predictor = MLPPredictor(12, 12, 128)  # bsz can be important ?
    elif data_args.predictor == 'hc':
        head_score_predictor = HardConcretePredictor(training_args, shape=(12, 12))
    else:
        raise Exception("Not supported head score predictor")
    optimizer = torch.optim.Adam(head_score_predictor.parameters(), lr=data_args.predictor_lr)
    head_score_predictor.to(training_args.device)
    l1_loss = SparseLoss()
    l1_loss.to(training_args.device)

    head_score = None
    for e in range(data_args.epoch_num):
        if head_score is not None:
            head_score = head_score.clone().detach()
        head_score = search_optimal_heads(training_args, model, head_score_predictor, optimizer, l1_loss,
                                          eval_dataloader,
                                          head_score, metric_name=data_args.metric_name,
                                          sparse_ratio=data_args.sparse_ratio,
                                          label_map=label_map)

    # compute corpus level head-importance for final head score prediction
    head_score = head_score.clone().detach()
    _, head_importance, preds, labels = compute_heads_importance(training_args, model, eval_dataloader,
                                                                 compute_entropy=False,
                                                                 compute_importance=True, head_mask=head_score,
                                                                 metric_name=data_args.metric_name)
    # head score for evaluation
    head_score_predictor.eval() 
    head_score = head_score_predictor(head_importance.transpose(1, 0))
    head_score = head_score.clone().detach()

    # prune_heads(args, model, eval_dataloader, head_mask)
    test_dataset = NerDataset(
        data_dir=data_args.data_dir,
        tokenizer=tokenizer,
        labels=labels,
        model_type=config.model_type,
        max_seq_length=data_args.max_seq_length,
        overwrite_cache=data_args.overwrite_cache,
        mode=Split.dev,
    )
    test_dataset.set_mode('half')
    test_dataset.set_index(1)  # use the other half

    test_sampler = RandomSampler(test_dataset) if training_args.local_rank == -1 else DistributedSampler(
        eval_dataset)
    test_dataloader = DataLoader(
        test_dataset, sampler=test_sampler, batch_size=training_args.train_batch_size,
        collate_fn=DefaultDataCollator().collate_batch
    )

    preds, labels = evaluate_masked_model(training_args, model, test_dataloader, head_score)
    final_score_dict = compute_metrics(EvalPrediction(preds, labels), label_map)
    with open(os.path.join(training_args.output_dir, 'mask_result.txt'), 'w') as f:
        logger.info("***** Eval results *****")
        for key, value in final_score_dict.items():
            logger.info("  %s = %s", key, value)
            f.write("%s = %s\n" % (key, value))
            f.write('remaining heads: %d\n' % head_score.sum())

    return final_score_dict


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
