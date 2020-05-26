#!/usr/bin/env python3
# Copyright 2018 CMU and The HuggingFace Inc. team.
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
""" Bertology: this script shows how you can explore the internals of the models in the library to:
    - compute the entropy of the head attentions
    - compute the importance of each head
    - prune (remove) the low importance head.
    Some parts of this script are adapted from the code of Michel et al. (http://arxiv.org/abs/1905.10650)
    which is available at https://github.com/pmichel31415/are-16-heads-really-better-than-1
"""
import argparse
import logging
import os
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Subset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DefaultDataCollator,
    GlueDataset,
    glue_compute_metrics,
    glue_output_modes,
    glue_processors,
    set_seed,
)

logger = logging.getLogger(__name__)


class RNNHeadPredictor(nn.Module):
    def __init__(self, head_num, layer_num, hidden_size, rnn_layers=2, bidirectional=True):
        super(RNNHeadPredictor, self).__init__()
        self.rnn = nn.LSTM(input_size=head_num, hidden_size=hidden_size,
                           num_layers=rnn_layers, bidirectional=bidirectional)
        self.linear = nn.Linear(hidden_size, head_num)
        self.act = nn.Sigmoid()
        self.layer_num = layer_num
        self.bidir = bidirectional
        self.hidden_size = hidden_size

    def forward(self, heads_importance):
        # heads_importance: layer_num, 1, head_num ( seq_len, bsz, input_size)
        heads_importance = heads_importance.unsqueeze(1)
        output, hiddens = self.rnn(heads_importance)
        if self.bidir:
            output = output[:, :, :self.hidden_size] + output[:, :, self.hidden_size:] # add forward and backward ret
        head_score = self.act(self.linear(output.squeeze()))  # seq_len, 1, head_num
        return head_score


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
        actually_pruned=False
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
    if not args.dont_normalize_importance_by_layer:
        exponent = 2
        norm_by_layer = torch.pow(torch.pow(head_importance, exponent).sum(-1), 1 / exponent)
        head_importance /= norm_by_layer.unsqueeze(-1) + 1e-20

    if not args.dont_normalize_global_importance:
        head_importance = (head_importance - head_importance.min()) / (head_importance.max() - head_importance.min())

    # Print/save matrices
    if compute_entropy:
        np.save(os.path.join(args.output_dir, "attn_entropy.npy"), attn_entropy.detach().cpu().numpy())
    if compute_importance:
        np.save(os.path.join(args.output_dir, "head_importance.npy"), head_importance.detach().cpu().numpy())

    #logger.info("Attention entropies")
    #print_2d_tensor(attn_entropy)
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


def search_optimal_heads(args, model, predictor, optimizer, sparse_loss, eval_dataloader, head_score):
    _, head_importance, preds, labels = compute_heads_importance(args, model, eval_dataloader, compute_entropy=False, head_mask=head_score)
    preds = np.argmax(preds, axis=1) if args.output_mode == "classification" else np.squeeze(preds)
    original_score = glue_compute_metrics(args.task_name, preds, labels)[args.metric_name]
    logger.info("Pruning: current  score: %f, threshold: %f", original_score, original_score * args.masking_threshold)
    head_score = predictor(head_importance)

    for step, inputs in enumerate(tqdm(eval_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])):
        for k, v in inputs.items():
            inputs[k] = v.to(args.device)
            # logger.info(k)

        # Do a forward pass (not with torch.no_grad() since we need gradients for updating predictor
        outputs = model(**inputs, head_mask=head_score)
        loss, logits, all_attentions = (
            outputs[0],
            outputs[1],
            outputs[-1],
        )  # Loss and logits are the first, attention the last
        s_loss = sparse_loss(head_score)
        total = loss  # +  s_loss  观察到一个现象 经常是同一个head 不同 Layer 都会置成0  不过，效果还是有一定的提升的
        total.backward()  # Backpropagate to populate the gradients in the head mask
        #logger.info('val loss: %f  sparse_loss : %f' %( loss.item(), s_loss.item() ) )
        optimizer.step()  # update predictor score
        head_score = predictor(head_importance)  # update head score
    logger.info('current total loss: %f' % total.item())
    logger.info('current head score')
    print_2d_tensor(head_score)
    return head_score 

def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models",
    )
    parser.add_argument(
        "--task_name",
        default=None,
        type=str,
        required=True,
        help="The name of the task to train selected in the list: " + ", ".join(glue_processors.keys()),
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    # Other parameters
    parser.add_argument(
        "--config_name",
        default="",
        type=str,
        help="Pretrained config name or path if not the same as model_name_or_path",
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name_or_path",
    )
    parser.add_argument(
        "--cache_dir",
        default=None,
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--data_subset", type=int, default=-1, help="If > 0: limit the data to a subset of data_subset instances."
    )
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Whether to overwrite data in output directory"
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )

    parser.add_argument(
        "--dont_normalize_importance_by_layer", action="store_true", help="Don't normalize importance score by layers"
    )
    parser.add_argument(
        "--dont_normalize_global_importance",
        action="store_true",
        help="Don't normalize all importance scores between 0 and 1",
    )


    parser.add_argument(
        "--masking_threshold",
        default=0.9,
        type=float,
        help="masking threshold in term of metrics (stop masking when metric < threshold * original metric value).",
    )
    parser.add_argument(
        "--masking_amount", default=0.1, type=float, help="Amount to heads to masking at each masking step."
    )
    parser.add_argument("--metric_name", default="acc", type=str, help="Metric to use for head masking.")

    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after WordPiece tokenization. \n"
             "Sequences longer than this will be truncated, sequences shorter padded.",
    )
    parser.add_argument("--batch_size", default=1, type=int, help="Batch size.")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--predictor_lr", type=float, default=1e-3)
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
    parser.add_argument("--no_cuda", action="store_true", help="Whether not to use CUDA when available")
    parser.add_argument("--server_ip", type=str, default="", help="Can be used for distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="Can be used for distant debugging.")
    args = parser.parse_args()

    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup devices and distributed training
    if args.local_rank == -1 or args.no_cuda:
        args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        args.device = torch.device("cuda", args.local_rank)
        args.n_gpu = 1
        torch.distributed.init_process_group(backend="nccl")  # Initializes the distributed backend

    # Setup logging
    logging.basicConfig(level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.info("device: {} n_gpu: {}, distributed: {}".format(args.device, args.n_gpu, bool(args.local_rank != -1)))

    # Set seeds
    set_seed(args.seed)

    # Prepare GLUE task
    args.task_name = args.task_name.lower()
    if args.task_name not in glue_processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = glue_processors[args.task_name]()
    args.output_mode = glue_output_modes[args.task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=args.task_name,
        output_attentions=True,
        cache_dir=args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path, cache_dir=args.cache_dir,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        cache_dir=args.cache_dir,
    )

    # Distributed and parallel training
    model.to(args.device)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )
    elif args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Print/save training arguments
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(args, os.path.join(args.output_dir, "run_args.bin"))
    logger.info("Training/evaluation parameters %s", args)

    # Prepare dataset for the GLUE task
    eval_dataset = GlueDataset(args, tokenizer=tokenizer, evaluate=True)
    if args.data_subset > 0:
        eval_dataset = Subset(eval_dataset, list(range(min(args.data_subset, len(eval_dataset)))))
    eval_sampler = RandomSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=args.batch_size, collate_fn=DefaultDataCollator().collate_batch
    )

    head_score_predictor = RNNHeadPredictor(12, 12, 32, rnn_layers=2)
    optimizer = torch.optim.SGD(head_score_predictor.parameters(), lr=args.predictor_lr)
    head_score_predictor.to(args.device)
    l1_loss = SparseLoss()
    l1_loss.to(args.device)
    head_score = None
    for e in range(10):
        if head_score is not None:
            head_score = head_score.clone().detach()
        head_score = search_optimal_heads(args, model, head_score_predictor, optimizer, l1_loss, eval_dataloader, head_score)
    # # Compute head entropy and importance score
    # #  compute_heads_importance(args, model, eval_dataloader)
    #
    # # Try head masking (set heads to zero until the score goes under a threshole)
    # # and head pruning (remove masked heads and see the effect on the network)
    # if args. and args.masking_threshold > 0.0 and args.masking_threshold < 1.0:
    #     head_mask = mask_heads(args, model, eval_dataloader)
    #     prune_heads(args, model, eval_dataloader, head_mask)


class SparseLoss(nn.Module):
    def __init__(self):
        super(SparseLoss, self).__init__()
        self.loss = nn.L1Loss()

    def forward(self, head_score):
        target = torch.zeros_like(head_score)
        return self.loss(head_score, target)


if __name__ == "__main__":
    main()
