# coding=utf-8

import dataclasses
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

import numpy as np

from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, EvalPrediction, GlueDataset, \
    LineByLineTextDataset, TextDataset, PreTrainedTokenizer, DataCollatorForLanguageModeling, AutoModelWithLMHead
from transformers.modeling_mixed import BranchyModel, math

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

    base_model_name_or_path: str = field(
        metadata={"help": "Path to pretrained large model or model identifier from huggingface.co/models"}
    )

    large_model_name_or_path: str = field(
        metadata={"help": "Path to pretrained base model or model identifier from huggingface.co/models"}
    )

    mixed_model_name_or_path: str = field(
        metadata={"help": "Path to pretrained base model or model identifier from huggingface.co/models"}
    )
    freeze_trained_models: bool = field(
        default=False,
        metadata={
            "help":
                "Freeze fine-tuned base and large models"},
    )

    iterative_training: bool = field(
        default=False,
        metadata={
            "help":
                "Iteratively train the switch path"},
    )

    kd_tl: bool = field(
        default=False,
        metadata={
            "help":
                "Knowledge Distillation Transformation Layers"},
    )

    only_cls: bool = field(
        default=False,
        metadata={
            "help":
                "Only Distill [CLS] representation when KD"},
    )

    freeze_classifiers: bool = field(
        default=False,
        metadata={
            "help":
                "freeze base & large model classifier"},
    )

    only_kd_loss: bool = field(
        default=False,
        metadata={
            "help":
                "Only fine-tune the TL with kd loss"},
    )

    share_tl: bool = field(
        default=False,
        metadata={
            "help":
                "Share transformation layers"},
    )

    unfreeze_large_encoder: bool = field(
        default=False,
        metadata={
            "help":
                "Freeze large encoder"},
    )

    unfreeze_base_encoder: bool = field(
        default=False,
        metadata={
            "help":
                "Freeze base encoder"},
    )

    non_linear_tl: bool = field(
        default=False,
        metadata={
            "help":
                "Non-linear TL layers"},
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

    saved_path: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )

    mode: Optional[str] = field(
        default="random", metadata={"help": "Use different mode for training or inference: [random, large, base]"}
    )
    base_model_handler: Optional[str] = field(
        default='bert',
        metadata={"help": "Large Model handler for underlying model in the SequenceClassification Model"}
    )
    large_model_handler: Optional[str] = field(
        default='bert',
        metadata={"help": "Large Model handler for underlying model in the SequenceClassification Model"}
    )

    entropy_threshold: Optional[float] = field(
        default=0.5,
        metadata={"help": "logits entropy threshold for switching "}
    )
    tl_kd_weight: Optional[float] = field(
        default=0.5,
        metadata={"help": "ratio of knowledge distillation loss of transformation layers"}
    )

    switch_pattern_idx: Optional[int] = field(
        default=-1,
        metadata={"help": "switch pattern idx"}
    )

    num_parts: Optional[int] = field(
        default=3,
        metadata={"help": "num of blocks for base and large models"}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    train_data_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a text file)."}
    )
    eval_data_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    line_by_line: bool = field(
        default=False,
        metadata={"help": "Whether distinct lines of text in the dataset are to be handled as distinct sequences."},
    )

    mlm: bool = field(
        default=False, metadata={"help": "Train with masked-language modeling loss instead of language modeling."}
    )
    mlm_probability: float = field(
        default=0.15, metadata={"help": "Ratio of tokens to mask for masked language modeling loss"}
    )

    block_size: int = field(
        default=-1,
        metadata={
            "help": "Optional input sequence length after tokenization."
                    "The training dataset will be truncated in block of this size for training."
                    "Default to the model max input length for single sentence inputs (take into account special tokens)."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )


def get_dataset(args: DataTrainingArguments, tokenizer: PreTrainedTokenizer, evaluate=False):
    file_path = args.eval_data_file if evaluate else args.train_data_file
    if args.line_by_line:
        return LineByLineTextDataset(tokenizer=tokenizer, file_path=file_path, block_size=args.block_size)
    else:
        return TextDataset(
            tokenizer=tokenizer, file_path=file_path, block_size=args.block_size, overwrite_cache=args.overwrite_cache
        )


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if data_args.eval_data_file is None and training_args.do_eval:
        raise ValueError(
            "Cannot do evaluation without an evaluation data file. Either supply a file to --eval_data_file "
            "or remove the --do_eval argument."
        )

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

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    print("--------------\n%s" % model_args.base_model_name_or_path)
    config_base = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.base_model_name_or_path,
        # num_labels=num_labels,
        # finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
    )

    config_large = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.large_model_name_or_path,
        # num_labels=num_labels,
        # finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.base_model_name_or_path,
        cache_dir=model_args.cache_dir,
    )

    model_base = AutoModelWithLMHead.from_pretrained(
        model_args.base_model_name_or_path,  # base model
        from_tf=bool(".ckpt" in model_args.base_model_name_or_path),
        config=config_base,
        cache_dir=model_args.cache_dir,
    )

    model_large = AutoModelWithLMHead.from_pretrained(
        model_args.large_model_name_or_path,  # large model
        from_tf=bool(".ckpt" in model_args.large_model_name_or_path),
        config=config_large,
        cache_dir=model_args.cache_dir,
    )

    if model_args.freeze_trained_models:
        if not model_args.unfreeze_base_encoder:
            logger.info("Freeze trained base models encoder ")
            for param in model_base.roberta.parameters():
                param.requires_grad = False
        if not model_args.unfreeze_large_encoder:
            logger.info("Freeze trained large models encoder ")
            for param in model_large.roberta.parameters():
                param.requires_grad = False

    if model_args.freeze_classifiers:
        logger.info("Freeze trained base & large models' classifier")
        for param in model_base.classifier.parameters():
            param.requires_grad = False
        for param in model_large.classifier.parameters():
            param.requires_grad = False

    if model_args.saved_path is not None:
        logger.info("Switch Pattern %d" % model_args.switch_pattern_idx)
        model = BranchyModel.from_pretrained(
            path=model_args.saved_path, model_base=model_base, model_large=model_large,
            switch_rate=0.5,
            num_parts=model_args.num_parts,
            base_model_name=model_args.base_model_handler,
            large_model_name=model_args.large_model_handler,
            entropy_threshold=model_args.entropy_threshold,
            switch_pattern_idx=model_args.switch_pattern_idx,
            share_tl=model_args.share_tl,
            tl_kd_weight=model_args.tl_kd_weight,
            only_cls=model_args.only_cls,
            only_kd_loss=model_args.only_kd_loss,
            non_linear_tl=model_args.non_linear_tl,
            pretrain_mlm=True,
            config=model_base.config)
    else:
        if model_args.switch_pattern_idx != -1:
            logger.info("Running switch pattern %d" % model_args.switch_pattern_idx)
        model = BranchyModel(model_base=model_base, model_large=model_large,
                             switch_rate=0.5,
                             num_parts=model_args.num_parts,
                             base_model_name=model_args.base_model_handler,
                             large_model_name=model_args.large_model_handler,
                             entropy_threshold=model_args.entropy_threshold,
                             switch_pattern_idx=model_args.switch_pattern_idx,
                             share_tl=model_args.share_tl,
                             kd_tl=model_args.kd_tl,
                             tl_kd_weight=model_args.tl_kd_weight,
                             only_cls=model_args.only_cls,
                             only_kd_loss=model_args.only_kd_loss,
                             non_linear_tl=model_args.non_linear_tl,
                             pretrain_mlm=True,
                             config=model_base.config)

    logger.info('len tokenizer: %d' %  len(tokenizer))
    model.model_base.resize_token_embeddings(len(tokenizer))
    model.model_large.resize_token_embeddings(len(tokenizer))

    # if config.model_type in ["bert", "roberta", "distilbert", "camembert"] and not data_args.mlm:
    #     raise ValueError(
    #         "BERT and RoBERTa-like models do not have LM heads but masked LM heads. They must be run using the --mlm "
    #         "flag (masked language modeling)."
    #     )

    if data_args.block_size <= 0:
        data_args.block_size = tokenizer.max_len
        # Our input block size will be the max possible for the model
    else:
        data_args.block_size = min(data_args.block_size, tokenizer.max_len)

    # Get datasets
    assert data_args.mlm, "must use mlm for training"

    train_dataset = get_dataset(data_args, tokenizer=tokenizer) if training_args.do_train else None
    eval_dataset = get_dataset(data_args, tokenizer=tokenizer, evaluate=True) if training_args.do_eval else None
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=data_args.mlm, mlm_probability=data_args.mlm_probability
    )

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        prediction_loss_only=True,
    )

    # Training
    if training_args.do_train:
        each_epoch_num = training_args.num_train_epochs
        if model_args.iterative_training:
            for pattern_idx in range(1, 7):
                model.set_pattern_idx(pattern_idx)
                trainer.train(model_path=model_args.mixed_model_name_or_path if os.path.isdir(
                    model_args.mixed_model_name_or_path) else None)
                trainer.args.num_train_epochs += each_epoch_num
        else:
            trainer.train(
                model_path=model_args.mixed_model_name_or_path if os.path.isdir(
                    model_args.mixed_model_name_or_path) else None
            )
            trainer.save_model()
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_master():
            tokenizer.save_pretrained(training_args.output_dir)

    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        eval_output = trainer.evaluate()

        perplexity = math.exp(eval_output["eval_loss"])
        result = {"perplexity": perplexity}

        output_eval_file = os.path.join(training_args.output_dir, "eval_results_lm.txt")
        if trainer.is_world_master():
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                    writer.write("%s = %s\n" % (key, str(result[key])))

        results.update(result)

    return results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
