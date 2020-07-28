# coding=utf-8

import dataclasses
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

import numpy as np

from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, EvalPrediction, GlueDataset
from transformers.modeling_mixed import BranchyModel

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

    share_tl: bool = field(
        default=False,
        metadata={
            "help":
            "Share transformation layers"},
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

    switch_pattern_idx: Optional[int] = field(
        default=-1,
        metadata={"help": "switch pattern idx"}
    )

    num_parts: Optional[int] = field(
        default=3,
        metadata={"help": "num of blocks for base and large models"}
    )


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

    print("--------------\n%s" % model_args.base_model_name_or_path)
    config_base = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.base_model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
    )

    config_large = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.large_model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.base_model_name_or_path,
        cache_dir=model_args.cache_dir,
    )

    model_base = AutoModelForSequenceClassification.from_pretrained(
        model_args.base_model_name_or_path,  # base model
        from_tf=bool(".ckpt" in model_args.base_model_name_or_path),
        config=config_base,
        cache_dir=model_args.cache_dir,
    )

    model_large = AutoModelForSequenceClassification.from_pretrained(
        model_args.large_model_name_or_path,  # large model
        from_tf=bool(".ckpt" in model_args.large_model_name_or_path),
        config=config_large,
        cache_dir=model_args.cache_dir,
    )

    if model_args.freeze_trained_models:
        logger.info("Freeze trained base & large models")
        for param in model_base.roberta.parameters():
            param.requires_grad = False
        for param in model_large.roberta.parameters():
            param.requires_grad = False

    if model_args.saved_path is not None:
        model = BranchyModel.from_pretrained(
            path=model_args.saved_path, model_base=model_base, model_large=model_large,
            switch_rate=0.5,
            base_model_name=model_args.base_model_handler,
            large_model_name=model_args.large_model_handler,
            entropy_threshold=model_args.entropy_threshold,
            switch_pattern_idx=model_args.switch_pattern_idx,
            share_tl=model_args.share_tl
        )
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
                             share_tl=model_args.share_tl)

    # Get datasets
    train_dataset = (
        GlueDataset(data_args, tokenizer=tokenizer, evaluate=False) if training_args.do_train else None
    )
    eval_dataset = (
        GlueDataset(data_args, tokenizer=tokenizer, evaluate=True)  # )mode="dev", cache_dir=model_args.cache_dir)
        if training_args.do_eval
        else None
    )
    test_dataset = (
        GlueDataset(data_args, tokenizer=tokenizer, evaluate=True)  # mode="test", cache_dir=model_args.cache_dir)
        if training_args.do_predict
        else None
    )

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

    # Evaluation
    eval_results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        eval_datasets = [eval_dataset]
        if data_args.task_name == "mnli":
            mnli_mm_data_args = dataclasses.replace(data_args, task_name="mnli-mm")
            # for acceraltion, we do not append mm dataset 
            #eval_datasets.append(
            #    GlueDataset(mnli_mm_data_args, tokenizer=tokenizer, evaluate=False)
                # mode="dev", cache_dir=model_args.cache_dir)
            #)

        for eval_dataset in eval_datasets:
            trainer.compute_metrics = build_compute_metrics_fn(eval_dataset.args.task_name)
            eval_result = trainer.evaluate(eval_dataset=eval_dataset)

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
