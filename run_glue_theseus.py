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
""" Finetuning the library models for sequence classification on GLUE (Bert, XLM, XLNet, RoBERTa, Albert, XLM-RoBERTa)."""
from copy import deepcopy

import dataclasses
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional
import time
import numpy as np
import torch.nn as nn

from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, EvalPrediction, GlueDataset, \
    AdamW, get_linear_schedule_with_warmup
from transformers.modeling_theseus_bert import BertForSequenceClassification, LinearReplacementScheduler, \
    ConstantReplacementScheduler, LinearPenaltyRatioScheduler, CurriculumLearningScheduler
from transformers import GlueDataTrainingArguments as DataTrainingArguments
from transformers import (
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    glue_compute_metrics,
    glue_output_modes,
    glue_tasks_num_labels,
    set_seed,
)

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    replacing_rate: float = field(
        metadata={"help": "Constant replacing rate. Also base replacing rate if using a scheduler."}

    )
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )

    scheduler_type: Optional[str] = field(
        default="none", metadata={"help": "Scheduler function"}
    )

    switch_pattern: Optional[int] = field(
        default=0, metadata={"help": "Switch pattern when inference, default 0 indicates scc layer"}
    )

    scheduler_linear_k: Optional[float] = field(
        default=0.0, metadata={"help": "linear k for replacement scheduler"}
    )

    steps_for_replacing: Optional[int] = field(
        default=0, metadata={"help": "Steps before entering successor fine_tuning (only useful for constant replacing"}
    )

    num_parts: Optional[int] = field(
        default=6, metadata={"help": "How many switches in base & large model"}
    )

    freeze_teacher: bool = field(
        default=False, metadata={"help": "freeze predecessor parameters, including layer, embedding, and output & "
                                         "pooler"}
    )

    fix_scc_layer: bool = field(
        default=False, metadata={"help": "fix  scc layer"}
    )

    early_exit: bool = field(
        default=False, metadata={"help": "add early exit branch"}
    )

    switch_mode: bool = field(
        default=False, metadata={"help": "Auto switch mode"}
    )

    train_early_exit: bool = field(
        default=False, metadata={"help": "first stage for training the early exit classifiers"}
    )

    init_highway: bool = field(
        default=False, metadata={"help": "open when first stage for training the early exit classifiers"}
    )

    use_baseline: bool = field(
        default=False, metadata={"help": "minus a baseline reward when PG"}
    )

    train_agent: bool = field(
        default=False, metadata={"help": "second stage for training the switch agent "}
    )

    only_large_and_exit: bool = field(
        default=False, metadata={"help": "Only large and exit action for switch agent"}
    )

    cl_scheduler: bool = field(
        default=False, metadata={"help": "use cl idx "}
    )

    cl_initial_idx: Optional[int] = field(
        default=5, metadata={"help": "initial_cl idx, default: num_parts -1 "}
    )

    cl_interval: Optional[int] = field(
        default=5, metadata={"help": "internal for decreasing cl idx"}
    )

    path_penalty_ratio: Optional[float] = field(
        default=0.0, metadata={"help": "path penalty for selecting large block"}
    )

    early_exit_idx: Optional[int] = field(
        default=-1, metadata={"help": "path penalty for selecting large block"}
    )

    logging_paths: bool = field(
        default=False, metadata={"help": "whether plotting paths learned during evaluating"}
    )

    pr_schedule: bool = field(
        default=False, metadata={"help": "use penalty ratio scheduler"}
    )

    pr_linear_k: Optional[float] = field(
        default=0.001, metadata={"help": "increase ratio for linear schedule penalty ratio"}
    )

    pr_init_value: Optional[float] = field(
        default=0.0, metadata={"help": "initial increase ratio for linear schedule penalty ratio"}
    )

    error_penalty: Optional[float] = field(
        default=0.0, metadata={"help": "error penalty when making wrong prediction"}
    )

    bound_alpha: Optional[float] = field(
        default=-1.0, metadata={"help": "bound alpha for adjusting the action probability"}
    )

    entropy_beta: Optional[float] = field(
        default=0.0, metadata={"help": "entropy weight for encouraging exploration"}
    )

    cl_idx: Optional[int] = field(
        default=-1, metadata={"help": "curriculum learning idx, denotes how many layer is set to large directly"}
    )


    #
    # parser.add_argument("--replacing_rate", type=float, required=True,
    #                     help="Constant replacing rate. Also base replacing rate if using a scheduler.")
    # parser.add_argument("--scheduler_type", default='none', choices=['none', 'linear'], help="Scheduler function.")
    # parser.add_argument("--scheduler_linear_k", default=0, type=float, help="Linear k for replacement scheduler.")
    # parser.add_argument("--steps_for_replacing", default=0, type=int,
    #                     help="Steps before entering successor fine_tuning (only useful for constant replacing)")


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

    try:
        num_labels = glue_tasks_num_labels[data_args.task_name]
        output_mode = glue_output_modes[data_args.task_name]
    except KeyError:
        raise ValueError("Task not found: %s" % (data_args.task_name))

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )
    model = BertForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir
    )

    model.set_entropy_beta(model_args.entropy_beta)

    if model_args.switch_pattern != 0:
        logger.info("Setting inference path as:%d " % model_args.switch_pattern)
        model.set_switch_pattern(model_args.switch_pattern)
    if model_args.num_parts != 6:
        logger.info("Setting num parts as: %d" % model_args.num_parts)
        model.bert.encoder.num_parts = model_args.num_parts

    # assert model_args.train_early_exit ^ model_args.train_agent, "Two stage can only train agent or early exit"
    if training_args.do_train:
        model.bert.encoder.init_agent_pooler(model.bert.pooler)  # init agent pooler

    if model_args.train_early_exit:
        # if second stage, the early exit is already trained
        model.bert.encoder.train_early_exit = True
        if model_args.init_highway:
            model.bert.init_highway_pooler()

    if model_args.train_agent:
        model.bert.encoder.train_agent = True  # using switch mode
        if model_args.only_large_and_exit:
            model.bert.encoder.only_large_and_exit = True

    if model_args.early_exit_idx != -1:
        logger.info("Setting early exit at block %d" % model_args.early_exit_idx)
        model.bert.encoder.early_exit_idx = model_args.early_exit_idx

    # Replace rate scheduler
    if model_args.scheduler_type == 'none':
        replacing_rate_scheduler = ConstantReplacementScheduler(bert_encoder=model.bert.encoder,
                                                                replacing_rate=model_args.replacing_rate,
                                                                replacing_steps=model_args.steps_for_replacing)
    elif model_args.scheduler_type == 'linear':
        replacing_rate_scheduler = LinearReplacementScheduler(bert_encoder=model.bert.encoder,
                                                              base_replacing_rate=model_args.replacing_rate,
                                                              k=model_args.scheduler_linear_k)
    else:
        raise ValueError("Unsupported scheduler type: %s" % model_args.scheduler_type)

    # error penalty
    if model_args.error_penalty != 0.0:
        logger.info("Setting error penalty to %.6f" % model_args.error_penalty)
        model.set_error_penalty(model_args.error_penalty)

    if model_args.bound_alpha != -1.0:
        logger.info("Setting bound alpha to %.6f" % model_args.bound_alpha)
        model.bert.encoder.bound_alpha = model_args.bound_alpha

    if model_args.cl_idx != -1:
        logger.info("Setting curriculum  learning idx to %d" % model_args.cl_idx)
        model.bert.encoder.cl_idx = model_args.cl_idx

    scc_n_layer = model.bert.encoder.scc_n_layer
    if training_args.do_train and not model_args.switch_mode:
        model.bert.encoder.scc_layer = nn.ModuleList(
            [deepcopy(model.bert.encoder.layer[ix]) for ix in range(scc_n_layer)])

    if model_args.pr_schedule:
        logger.info("linearly increase penalty ratio from %.6f to  %.6f with linear_k %.6f" % (model_args.pr_init_value,
                                                                                               model_args.path_penalty_ratio,
                                                                                               model_args.pr_linear_k))
        pr_scheduler = LinearPenaltyRatioScheduler(model,
                                                   initial_penalty_ratio=model_args.pr_init_value,
                                                   linear_k=model_args.pr_linear_k,
                                                   max_penalty_ratio=model_args.path_penalty_ratio)
    else:
        if model_args.path_penalty_ratio > 0:
            logger.info("setting path penalty to: %.6f" % model_args.path_penalty_ratio)
        pr_scheduler = None
        model.set_path_penalty(model_args.path_penalty_ratio)
    # if model_args.freeze_teacher:
    #     for p in model.bert.encoder.layer.parameters():
    #         p.requires_grad = False
    #     for p in model.bert.embeddings.parameters():
    #         p.requires_grad = False
    #     for p in model.bert.pooler.parameters():
    #         p.requires_grad = False
    #     for p in model.classifier:
    #         p.requires_grad = False
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = []

    if model_args.use_baseline:
        model.set_baseline()

    if not model_args.fix_scc_layer:
        optimizer_grouped_parameters.extend([
            {'params': [p for n, p in model.bert.encoder.scc_layer.named_parameters() if
                        not any(nd in n for nd in no_decay)], 'weight_decay': training_args.weight_decay},
            {'params': [p for n, p in model.bert.encoder.scc_layer.named_parameters() if
                        any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        ])
    if model_args.switch_mode:
        # we first train early exit, than train the agent ?
        if model_args.train_early_exit:
            optimizer_grouped_parameters.extend([
                # {'params': [p for p in model.bert.encoder.agent.parameters()]},
                {
                    'params': [p for p in model.bert.encoder.early_classifiers.parameters()],
                    "weight_decay": training_args.weight_decay
                }
            ]
            )

        elif model_args.train_agent:
            optimizer_grouped_parameters.extend([
                {'params': [p for p in model.bert.encoder.agent.parameters()],
                 "weight_decay": training_args.weight_decay
                 },
                {'params': [p for p in model.bert.encoder.simple_agent.parameters()],
                 "weight_decay": training_args.weight_decay
                 },
                # {'params': [p for p in model.bert.encoder.early_classifiers.parameters()]}
            ])
        else:
            optimizer_grouped_parameters.extend([
                {'params': [p for p in model.bert.encoder.agent.parameters()]},
                {'params': [p for p in model.bert.encoder.early_classifiers.parameters()]}
            ])
    if model_args.cl_scheduler:
        cl_scheduler = CurriculumLearningScheduler(model.bert.encoder,
                                                   initial_cl_idx=model_args.num_parts - 1,  # default
                                                   epoch_interval=model_args.cl_interval)
    else:
        cl_scheduler = None

        # Get datasets
    train_dataset = (
        GlueDataset(data_args, tokenizer=tokenizer, evaluate=False) if training_args.do_train else None
    )
    eval_dataset = (
        GlueDataset(data_args, tokenizer=tokenizer, evaluate=True)  # mode="dev", cache_dir=model_args.cache_dir)
        if training_args.do_eval
        else None
    )
    test_dataset = (
        GlueDataset(data_args, tokenizer=tokenizer, evaluate=True)  #
        if training_args.do_predict
        else None
    )

    # train_dataset = eval_dataset # for testing agent capability
    def build_compute_metrics_fn(task_name: str) -> Callable[[EvalPrediction], Dict]:
        def compute_metrics_fn(p: EvalPrediction):
            if output_mode == "classification":
                preds = np.argmax(p.predictions, axis=1)
            elif output_mode == "regression":
                preds = np.squeeze(p.predictions)
            return glue_compute_metrics(task_name, preds, p.label_ids)

        return compute_metrics_fn

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=build_compute_metrics_fn(data_args.task_name),
        theseus_replace_scheduler=replacing_rate_scheduler,
        optimizer_grouped_parameters=optimizer_grouped_parameters,
        logging_paths=model_args.logging_paths,
        pr_scheduler=pr_scheduler,
        cl_scheduler=cl_scheduler
    )

    # Training
    if training_args.do_train:
        trainer.train(
            model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
        )
        trainer.save_model()
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_master():
            tokenizer.save_pretrained(training_args.output_dir)

    # Evaluation
    eval_results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        eval_datasets = [eval_dataset]
        # if data_args.task_name == "mnli":
        #     mnli_mm_data_args = dataclasses.replace(data_args, task_name="mnli-mm")
        #     eval_datasets.append(
        #         GlueDataset(mnli_mm_data_args, tokenizer=tokenizer, mode="dev", cache_dir=model_args.cache_dir)
        #     )

        for eval_dataset in eval_datasets:
            trainer.compute_metrics = build_compute_metrics_fn(eval_dataset.args.task_name)
            eval_result = trainer.evaluate(eval_dataset=eval_dataset, require_paths=model_args.logging_paths)
            output_eval_file = os.path.join(
                training_args.output_dir, f"eval_results_{eval_dataset.args.task_name}.txt"
            )
            if trainer.is_world_master():
                with open(output_eval_file, "w") as writer:
                    logger.info("***** Eval results {} *****".format(eval_dataset.args.task_name))
                    for key, value in eval_result.items():
                        logger.info("  %s = %s", key, value)
                        writer.write("%s = %s\n" % (key, value))
            eval_results.update(eval_result)

    if training_args.do_predict:
        logging.info("*** Test ***")
        test_datasets = [test_dataset]
        if data_args.task_name == "mnli":
            mnli_mm_data_args = dataclasses.replace(data_args, task_name="mnli-mm")
            test_datasets.append(
                GlueDataset(mnli_mm_data_args, tokenizer=tokenizer, mode="test", cache_dir=model_args.cache_dir)
            )

        for test_dataset in test_datasets:
            predictions = trainer.predict(test_dataset=test_dataset).predictions
            if output_mode == "classification":
                predictions = np.argmax(predictions, axis=1)

            output_test_file = os.path.join(
                training_args.output_dir, f"test_results_{test_dataset.args.task_name}.txt"
            )
            if trainer.is_world_master():
                with open(output_test_file, "w") as writer:
                    logger.info("***** Test results {} *****".format(test_dataset.args.task_name))
                    writer.write("index\tprediction\n")
                    for index, item in enumerate(predictions):
                        if output_mode == "regression":
                            writer.write("%d\t%3.3f\n" % (index, item))
                        else:
                            item = test_dataset.get_labels()[item]
                            writer.write("%d\t%s\n" % (index, item))
    return eval_results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
