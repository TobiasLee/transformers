import logging
import os
import time
from dataclasses import dataclass, field
from typing import List, Optional

import torch
from filelock import FileLock
from torch.utils.data.dataset import Dataset

from ...tokenization_roberta import RobertaTokenizer, RobertaTokenizerFast
from ...tokenization_utils import PreTrainedTokenizer
from ...tokenization_xlm_roberta import XLMRobertaTokenizer
from ..processors.glue import glue_convert_examples_to_features, glue_output_modes, glue_processors
from ..processors.utils import InputFeatures

logger = logging.getLogger(__name__)


@dataclass
class GlueDataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    task_name: str = field(metadata={"help": "The name of the task to train on: " + ", ".join(glue_processors.keys())})
    data_dir: str = field(
        metadata={"help": "The input data dir. Should contain the .tsv files (or other data files) for the task."}
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

    def __post_init__(self):
        self.task_name = self.task_name.lower()


class GlueDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    args: GlueDataTrainingArguments
    output_mode: str
    features: List[InputFeatures]

    def __init__(
            self,
            args: GlueDataTrainingArguments,
            tokenizer: PreTrainedTokenizer,
            limit_length: Optional[int] = None,
            evaluate=False,
    ):
        self.args = args
        processor = glue_processors[args.task_name]()
        self.label_list = processor.get_labels()

        self.output_mode = glue_output_modes[args.task_name]
        self.task_label_list = processor.get_task_labels() if self.output_mode =='multitask' else None 
        # Load data features from cache or dataset file
        cached_features_file = os.path.join(
            args.data_dir,
            "cached_{}_{}_{}_{}".format(
                "dev" if evaluate else "train", tokenizer.__class__.__name__, str(args.max_seq_length), args.task_name,
            ),
        )

        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        lock_path = cached_features_file + ".lock"
        with FileLock(lock_path):

            if os.path.exists(cached_features_file) and not args.overwrite_cache:
                start = time.time()
                self.features = torch.load(cached_features_file)
                self.features_for_head_pruning = self.features[: len(self.features) // 2]
                self.features_for_dev = self.features[len(self.features) // 2:]
                logger.info(
                    f"Loading features from cached file {cached_features_file} [took %.3f s]", time.time() - start
                )
            else:
                logger.info(f"Creating features from dataset file at {args.data_dir}")
                label_list = processor.get_labels()
                if args.task_name in ["mnli", "mnli-mm"] and tokenizer.__class__ in (
                        RobertaTokenizer,
                        RobertaTokenizerFast,
                        XLMRobertaTokenizer,
                ):
                    # HACK(label indices are swapped in RoBERTa pretrained model)
                    label_list[1], label_list[2] = label_list[2], label_list[1]
                examples = (
                    processor.get_dev_examples(args.data_dir)
                    if evaluate
                    else processor.get_train_examples(args.data_dir)
                )
                if limit_length is not None:
                    examples = examples[:limit_length]
                self.features = glue_convert_examples_to_features(
                    examples,
                    tokenizer,
                    max_length=args.max_seq_length,
                    label_list=label_list,
                    task_label_list=self.task_label_list,
                    output_mode=self.output_mode,
                )
                logger.info("creating half features")
                self.features_for_head_pruning = self.features[: len(self.features) // 2]
                self.features_for_dev = self.features[len(self.features) // 2:]
                start = time.time()
                torch.save(self.features, cached_features_file)
                # ^ This seems to take a lot of time so I want to investigate why and how we can improve.
                logger.info(
                    "Saving features into cached file %s [took %.3f s]", cached_features_file, time.time() - start
                )
        self.mode = 'all'
        self.current_idx = 0

    def __len__(self):
        if self.mode == "all":
            return len(self.features)  # do not split
        elif self.mode == 'half':  # use half to searching optimal architecture
            if self.current_idx == 0:  # half
                return len(self.features_for_head_pruning)
            elif self.current_idx == 1:
                return len(self.features_for_dev)

    def __getitem__(self, i):
        if self.mode == "all":
            return self.features[i]  # do not split
        elif self.mode == 'half':  # use half to searching optimal architecture
            if self.current_idx == 0:  # half
                return self.features_for_head_pruning[i]
            elif self.current_idx == 1:
                return self.features_for_dev[i]

    def set_mode(self, mode='all'):
        assert mode in ['all', 'half'], "Only support half or all mode"
        self.mode = mode

    def set_index(self, index):
        self.current_idx = index
    
    def get_labels(self):
        return self.label_list
