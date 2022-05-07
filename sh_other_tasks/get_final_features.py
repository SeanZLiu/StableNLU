"""
Script to train BERT on MNLI with our loss function

Modified from the old "run_classifier" script from
https://github.com/huggingface/pytorch-transformer
"""
import pickle
import argparse
import json
import logging
import os
import random
from collections import namedtuple
from os.path import join, exists
from typing import List, Dict, Iterable

# temporary hack for the pythonroot issue
import sys
from sklearn.metrics import f1_score
import numpy as np
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.modeling import BertConfig, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.modeling import BertForSequenceClassification
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear
from pytorch_pretrained_bert.tokenization import BertTokenizer
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Dataset, \
    Sampler
from tqdm import trange, tqdm

import config
import utils

import clf_distill_loss_functions
from bert_distill import BertDistill
from clf_distill_loss_functions import *

from predictions_analysis import visualize_predictions
from utils import Processor, process_par

# Its a hack, but I didn't want a tensorflow dependency be required to run this code, so
# for now we just copy-paste MNLI loading stuff from debias/datasets/mnli

HANS_URL = "https://raw.githubusercontent.com/tommccoy1/hans/master/heuristics_evaluation_set.txt"

NLI_LABELS = ["contradiction", "entailment", "neutral"]
NLI_LABEL_MAP = {k: i for i, k in enumerate(NLI_LABELS)}
REV_NLI_LABEL_MAP = {i: k for i, k in enumerate(NLI_LABELS)}
NLI_LABEL_MAP["hidden"] = NLI_LABEL_MAP["entailment"]

FEVER_LABELS = ['REFUTES','SUPPORTS','NOT ENOUGH INFO']
FEVER_LABEL_MAP = {"SUPPORTS": 0, "REFUTES": 1, "NOT ENOUGH INFO": 2}

TextPairExample = namedtuple("TextPairExample", ["id", "premise", "hypothesis", "label"])

def load_paws(is_train,n_samples=None) -> List[
    TextPairExample]:
    out = []
    if is_train:
        filename = join(config.QQP_PAWS_SOURCE, "train.tsv")
    else:
        filename = join(config.QQP_PAWS_SOURCE, "dev_and_test.tsv")
    with open(filename, "r") as f:
        f.readline()
        lines = f.readlines()

    if n_samples is not None:
        lines = np.random.RandomState(16349 + n_samples).choice(lines, n_samples,
                                                                replace=False)
    for line in lines:
        line = line.split("\t")
        out.append(
            TextPairExample(line[0], line[1][2:-1], line[2][2:-1], int(line[-1])))
    return out

def load_Symmetric(n_samples=None) -> List[
    TextPairExample]:
    out = []
    filename = join(config.FEVER_SOURCE, "fever_symmetric_generated.jsonl")  # 注意symmetric和fever放在同一目录下
    with open(filename, "r") as f:
        f.readline()
        lines = f.readlines()

    if n_samples is not None:
        lines = np.random.RandomState(16349 + n_samples).choice(lines, n_samples,
                                                                replace=False)
    for line in lines:
        example = json.loads(line)
        out.append(
            TextPairExample(str(example['id']), example['claim'], example['evidence_sentence'], FEVER_LABEL_MAP[example['label'].rstrip()]))
        # 这里的label需要将字符串map为数字

    return out

def load_Symmetric_test(n_samples=None) -> List[
    TextPairExample]:
    out = []
    filename = join(config.FEVER_SOURCE, "fever_symmetric_test.jsonl")  # 注意symmetric和fever放在同一目录下
    with open(filename, "r") as f:
        f.readline()
        lines = f.readlines()

    if n_samples is not None:
        lines = np.random.RandomState(16349 + n_samples).choice(lines, n_samples,
                                                                replace=False)
    for line in lines:
        example = json.loads(line)
        out.append(
            TextPairExample(int(example['id']), example['claim'], example['evidence'], FEVER_LABEL_MAP[example['gold_label'].rstrip()]))
        # 这里的label需要将字符串map为数字

    return out

def load_Symmetric_full(n_samples=None) -> List[
    TextPairExample]:
    out = []
    filename = join(config.FEVER_SOURCE, "fever_symmetric_full.jsonl")  # 注意symmetric和fever放在同一目录下
    with open(filename, "r") as f:
        f.readline()
        lines = f.readlines()

    if n_samples is not None:
        lines = np.random.RandomState(16349 + n_samples).choice(lines, n_samples,
                                                                replace=False)
    for line in lines:
        example = json.loads(line)
        out.append(
            TextPairExample(int(example['id']), example['claim'], example['evidence_sentence'], FEVER_LABEL_MAP[example['label'].rstrip()]))
        # 这里的label需要将字符串map为数字

    return out

def load_fever(is_train, sample=None, custom_path=None) -> List[TextPairExample]:
    if is_train:
        filename = join(config.FEVER_SOURCE, "fever_train.jsonl")
    else:
        if custom_path is None:
            filename = join(config.FEVER_SOURCE, "fever_dev.jsonl")
        else:
            filename = join(config.FEVER_SOURCE, custom_path)

    logging.info("Loading fever " + ("train" if is_train else "dev"))
    with open(filename) as f:
        f.readline()
        lines = f.readlines()

    if sample:
        lines = np.random.RandomState(26096781 + sample).choice(lines, sample,
                                                                replace=False)

    out = []
    for line in lines:
        example = json.loads(line)
        out.append(
            TextPairExample(str(example['id']), example['claim'], example['evidence'], FEVER_LABEL_MAP[example['gold_label'].rstrip()]))
    return out

def load_qqp(is_train, sample=None, custom_path=None) -> List[TextPairExample]:
    # 直接读取原始qqp的train.tsv或dev.tsv，不需要再将文件格式进行转变了
    # ensure qqp is downloaded ?
    if is_train:
        filename = join(config.QQP_SOURCE, "train.tsv")
    else:
        if custom_path is None:
            filename = join(config.QQP_SOURCE, "dev.tsv")
        else:
            filename = join(config.QQP_SOURCE, custom_path)

    # logging.info("Loading mnli " + ("train" if is_train else "dev"))
    logging.info("Loading qqp " + ("train" if is_train else "dev"))

    with open(filename) as f:
        f.readline()
        lines = f.readlines()

    if sample:
        lines = np.random.RandomState(26096781 + sample).choice(lines, sample,
                                                                replace=False)

    out = []
    for line in lines:
        line = line.split("\t")
        if len(line) != 6:
            continue
        out.append(
            TextPairExample(line[0], line[3], line[4], int(line[-1])))
    return out


def load_easy_hard(prefix="", no_mismatched=False):
    all_datasets = []

    all_datasets.append(("mnli_dev_matched_{}easy".format(prefix),
                         load_mnli(False, custom_path="dev_matched_{}easy.tsv".format(prefix))))
    all_datasets.append(("mnli_dev_matched_{}hard".format(prefix),
                         load_mnli(False, custom_path="dev_matched_{}hard.tsv".format(prefix))))
    if not no_mismatched:
        all_datasets.append(("mnli_dev_mismatched_{}easy".format(prefix),
                             load_mnli(False, custom_path="dev_mismatched_{}easy.tsv".format(prefix))))
        all_datasets.append(("mnli_dev_mismatched_{}hard".format(prefix),
                             load_mnli(False, custom_path="dev_mismatched_{}hard.tsv".format(prefix))))

    return all_datasets


def load_hans_subsets():
    src = join(config.HANS_SOURCE, "heuristics_evaluation_set.txt")
    if not exists(src):
        logging.info("Downloading source to %s..." % config.HANS_SOURCE)
        utils.download_to_file(HANS_URL, src)

    hans_datasets = []
    labels = ["entailment", "non-entailment"]
    subsets = set()
    with open(src, "r") as f:
        for line in f.readlines()[1:]:
            line = line.split("\t")
            subsets.add(line[-3])
    subsets = [x for x in subsets]

    for label in labels:
        for subset in subsets:
            name = "hans_{}_{}".format(label, subset)
            examples = load_hans(filter_label=label, filter_subset=subset)
            hans_datasets.append((name, examples))

    return hans_datasets


def load_hans(n_samples=None, filter_label=None, filter_subset=None) -> List[
    TextPairExample]:
    out = []

    if filter_label is not None and filter_subset is not None:
        logging.info("Loading hans subset: {}-{}...".format(filter_label, filter_subset))
    else:
        logging.info("Loading hans all...")

    src = join(config.HANS_SOURCE, "heuristics_evaluation_set.txt")
    if not exists(src):
        logging.info("Downloading source to %s..." % config.HANS_SOURCE)
        utils.download_to_file(HANS_URL, src)

    with open(src, "r") as f:
        f.readline()
        lines = f.readlines()

    if n_samples is not None:
        lines = np.random.RandomState(16349 + n_samples).choice(lines, n_samples,
                                                                replace=False)

    for line in lines:
        parts = line.split("\t")
        label = parts[0]

        if filter_label is not None and filter_subset is not None:
            if label != filter_label or parts[-3] != filter_subset:
                continue

        if label == "non-entailment":
            label = 0
        elif label == "entailment":
            label = 1
        else:
            raise RuntimeError()
        s1, s2, pair_id = parts[5:8]
        out.append(TextPairExample(pair_id, s1, s2, label))
    return out


def ensure_mnli_is_downloaded():
    mnli_source = config.GLUE_SOURCE
    if exists(mnli_source) and len(os.listdir(mnli_source)) > 0:
        return
    else:
        raise Exception("Download MNLI from Glue and put files under glue_multinli")


def load_mnli(is_train, sample=None, custom_path=None) -> List[TextPairExample]:
    ensure_mnli_is_downloaded()
    if is_train:
        filename = join(config.GLUE_SOURCE, "train.tsv")
    else:
        if custom_path is None:
            filename = join(config.GLUE_SOURCE, "dev_matched.tsv")
        else:
            filename = join(config.GLUE_SOURCE, custom_path)

    logging.info("Loading mnli " + ("train" if is_train else "dev"))
    with open(filename) as f:
        f.readline()
        lines = f.readlines()

    if sample:
        lines = np.random.RandomState(26096781 + sample).choice(lines, sample,
                                                                replace=False)

    out = []
    for line in lines:
        line = line.split("\t")
        out.append(
            TextPairExample(line[0], line[8], line[9], NLI_LABEL_MAP[line[-1].rstrip()]))
    return out


def load_teacher_probs(custom_teacher=None):
    if custom_teacher is None:
        file_path = config.TEACHER_SOURCE
    else:
        file_path = custom_teacher

    with open(file_path, "r") as teacher_file:
        all_lines = teacher_file.read()
        all_json = json.loads(all_lines)

    return all_json


def load_bias(bias_name, custom_path=None) -> Dict[str, np.ndarray]:
    """Load dictionary of example_id->bias where bias is a length 3 array
    of log-probabilities"""

    if custom_path is not None:  # file contains probs
        with open(custom_path, "r") as bias_file:
            all_lines = bias_file.read()
            bias = json.loads(all_lines)
            for k, v in bias.items():
                bias[k] = np.log(np.array(v))
        return bias

    if bias_name == "hans":
        if bias_name == "hans":
            bias_src = config.MNLI_WORD_OVERLAP_BIAS
        if not exists(bias_src):
            raise Exception("lexical overlap bias file is not found")
        bias = utils.load_pickle(bias_src)
        for k, v in bias.items():
            # Convert from entail vs non-entail to 3-way classes by splitting non-entail
            # to neutral and contradict
            bias[k] = np.array([
                v[0] - np.log(2.),
                v[1],
                v[0] - np.log(2.),
            ])
        return bias

    if bias_name in config.BIAS_SOURCES:
        file_path = config.BIAS_SOURCES[bias_name]
        with open(file_path, "r") as hypo_file:
            all_lines = hypo_file.read()
            bias = json.loads(all_lines)
            for k, v in bias.items():
                bias[k] = np.array(v)
        return bias
    else:
        raise Exception("invalid bias name")


def load_all_test_jsonl():
    test_datasets = []
    test_datasets.append(("mnli_test_m", load_jsonl("multinli_0.9_test_matched_unlabeled.jsonl",
                                                    config.MNLI_TEST_SOURCE)))
    test_datasets.append(("mnli_test_mm", load_jsonl("multinli_0.9_test_mismatched_unlabeled.jsonl",
                                                     config.MNLI_TEST_SOURCE)))
    test_datasets.append(("mnli_test_hard_m", load_jsonl("multinli_0.9_test_matched_unlabeled_hard.jsonl",
                                                         config.MNLI_HARD_SOURCE)))
    test_datasets.append(("mnli_test_hard_mm", load_jsonl("multinli_0.9_test_mismatched_unlabeled_hard.jsonl",
                                                          config.MNLI_HARD_SOURCE)))
    return test_datasets


def load_jsonl(file_path, data_dir, sample=None):
    out = []
    full_path = join(data_dir, file_path)
    logging.info("Loading jsonl from {}...".format(full_path))
    with open(full_path, 'r') as jsonl_file:
        for i, line in enumerate(jsonl_file):
            example = json.loads(line)

            label = example["gold_label"]
            if label == '-':
                continue

            if not "pairID" in example.keys():
                id = i
            else:
                id = example["pairID"]
            text_a = example["sentence1"]
            text_b = example["sentence2"]

            out.append(TextPairExample(id, text_a, text_b, NLI_LABEL_MAP[label]))

    if sample:
        random.shuffle(out)
        out = out[:sample]

    return out


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, example_id, input_ids, segment_ids, label_id, bias, bias_features):
        self.example_id = example_id
        self.input_ids = input_ids
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.bias = bias
        # todo bias_features为10 * 768 的FloatTensor
        self.bias_features = bias_features
        # todo


class ExampleConverter(Processor):
    def __init__(self, max_seq_length, tokenizer, shallow_features):
        self.max_seq_length = max_seq_length
        self.tokenizer = tokenizer
        # todo
        self.shallow_features = shallow_features
        # todo

    def process(self, data: Iterable):
        features = []
        tokenizer = self.tokenizer
        max_seq_length = self.max_seq_length

        # todo shallow features  final
        shallow_features = self.shallow_features
        # todo

        for example in data:
            tokens_a = tokenizer.tokenize(example.premise)

            tokens_b = None
            if example.hypothesis:
                tokens_b = tokenizer.tokenize(example.hypothesis)
                # Modshallow_featuresself.bias_featuresifies `tokens_a` and `tokens_b` in place so that the total
                # length is less than the specified length.
                # Account for [CLS], [SEP], [SEP] with "- 3"
                _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
            else:
                # Account for [CLS] and [SEP] with "- 2"
                if len(tokens_a) > max_seq_length - 2:
                    tokens_a = tokens_a[:(max_seq_length - 2)]

            tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
            segment_ids = [0] * len(tokens)

            if tokens_b:
                tokens += tokens_b + ["[SEP]"]
                segment_ids += [1] * (len(tokens_b) + 1)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # todo add shallow features     final
            shallow_fea = shallow_features[example.id] if example.id in shallow_features.keys() else shallow_features[int(example.id)]
            shallow_fea = torch.Tensor(shallow_fea)
            #print(example.id)
            #print(len(shallow_fea[0]))
            #print(len(shallow_fea))
            features.append(
                InputFeatures(
                    example_id=example.id,
                    input_ids=np.array(input_ids),
                    segment_ids=np.array(segment_ids),
                    label_id=example.label,
                    bias=None,
                    bias_features=shallow_fea
                ))
            # todo

        return features


class InputFeatureDataset(Dataset):

    def __init__(self, examples: List[InputFeatures]):
        self.examples = examples

    def __getitem__(self, index):
        return self.examples[index]

    def __len__(self):
        return len(self.examples)


def collate_input_features(batch: List[InputFeatures]):
    max_seq_len = max(len(x.input_ids) for x in batch)
    sz = len(batch)

    input_ids = np.zeros((sz, max_seq_len), np.int64)
    segment_ids = np.zeros((sz, max_seq_len), np.int64)
    mask = torch.zeros(sz, max_seq_len, dtype=torch.int64)
    for i, ex in enumerate(batch):
        input_ids[i, :len(ex.input_ids)] = ex.input_ids
        segment_ids[i, :len(ex.segment_ids)] = ex.segment_ids
        mask[i, :len(ex.input_ids)] = 1

    input_ids = torch.as_tensor(input_ids)
    segment_ids = torch.as_tensor(segment_ids)
    label_ids = torch.as_tensor(np.array([x.label_id for x in batch], np.int64))

    # include example ids for test submission
    try:
        example_ids = torch.tensor([int(x.example_id) for x in batch])
    except:
        example_ids = torch.zeros(len(batch)).long()

    if batch[0].bias is None:
        return example_ids, input_ids, mask, segment_ids, label_ids

    teacher_probs = torch.tensor([x.teacher_probs for x in batch])
    bias = torch.tensor([x.bias for x in batch])

    return example_ids, input_ids, mask, segment_ids, label_ids, bias, teacher_probs


class SortedBatchSampler(Sampler):
    def __init__(self, data_source, batch_size, seed):
        super().__init__(data_source)
        self.data_source = data_source
        self.batch_size = batch_size
        self.seed = seed
        if batch_size == 1:
            raise NotImplementedError()
        self._epoch = 0

    def __iter__(self):
        rng = np.random.RandomState(self._epoch + 601767 + self.seed)
        n_batches = len(self)
        batch_lens = np.full(n_batches, self.batch_size, np.int32)

        # Randomly select batches to reduce by size 1
        extra = n_batches * self.batch_size - len(self.data_source)
        batch_lens[rng.choice(len(batch_lens), extra, False)] -= 1

        batch_ends = np.cumsum(batch_lens)
        batch_starts = np.pad(batch_ends[:-1], [1, 0], "constant")

        if batch_ends[-1] != len(self.data_source):
            print(batch_ends)
            raise RuntimeError()

        bounds = np.stack([batch_starts, batch_ends], 1)
        rng.shuffle(bounds)

        for s, e in bounds:
            yield np.arange(s, e)

    def __len__(self):
        return (len(self.data_source) + self.batch_size - 1) // self.batch_size


def build_train_dataloader(data: List[InputFeatures], batch_size, seed, sorted):
    if sorted:
        data.sort(key=lambda x: len(x.input_ids))
        ds = InputFeatureDataset(data)
        sampler = SortedBatchSampler(ds, batch_size, seed)
        return DataLoader(ds, batch_sampler=sampler, collate_fn=collate_input_features)
    else:
        ds = InputFeatureDataset(data)
        return DataLoader(ds, batch_size=batch_size, sampler=RandomSampler(ds),
                          collate_fn=collate_input_features)


def build_eval_dataloader(data: List[InputFeatures], batch_size):
    ds = InputFeatureDataset(data)
    return DataLoader(ds, batch_size=batch_size, sampler=SequentialSampler(ds),
                      collate_fn=collate_input_features)

# todo have changed
def convert_examples_to_features(
        examples: List[TextPairExample], max_seq_length, tokenizer,shallow_features, n_process=1):
    converter = ExampleConverter(max_seq_length, tokenizer,shallow_features=shallow_features)
    return process_par(examples, converter, n_process, chunk_size=2000, desc="featurize")
# todo

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def simple_accuracy(preds, labels):
    return (preds == labels).mean()

def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    f1_non = f1_score(y_true=labels, y_pred=preds, pos_label=0)
    return {
        "acc": acc,
        "f1": f1,
        "f1_non": f1_non
    }

def main():
    parser = argparse.ArgumentParser()

  
    parser.add_argument("--task", default="mnli", type=str,help="mnli or fever or qqp")
    parser.add_argument("--get_bert_output", action='store_true',default=False)
    parser.add_argument("--gene_challenge",action='store_true',default=False)
    parser.add_argument("--shallow_feature_file", type=str, help="the pkl file to save the shallow feature.")
    parser.add_argument("--final_feature",type=str,help="the pkl file to save final features ,which have already fused "
                                                        "with shallow feature")



    # todo 选择任务
    ## Required parameters


    parser.add_argument("--bert_model", default="bert-base-uncased", type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                             "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--cache_dir",
                        default="",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_eval_on_train",
                        action='store_true',
                        help="Whether to run eval on the train set.")
    parser.add_argument("--do_test",
                        action='store_true',
                        help="Whether to run test and create submission.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--seed",
                        default=None,
                        type=int,
                        help="Seed for randomized elements in the training")
    parser.add_argument("--eval_batch_size",
                        default=16,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")

    ## Our arguements
    parser.add_argument("--mode", choices=["none", "distill", "smoothed_distill", "smoothed_distill_annealed",
                                           "label_smoothing", "theta_smoothed_distill", "reweight_baseline",
                                           "smoothed_reweight_baseline", "permute_smoothed_distill",
                                           "bias_product_baseline", "learned_mixin_baseline",
                                           "reweight_by_teacher", "reweight_by_teacher_annealed",
                                           "bias_product_by_teacher", "bias_product_by_teacher_annealed",
                                           "focal_loss"])
    parser.add_argument("--penalty", type=float, default=0.03,
                        help="Penalty weight for the learn_mixin model")
    parser.add_argument("--focal_loss_gamma", type=float, default=1.0)
    parser.add_argument("--n_processes", type=int, default=4,
                        help="Processes to use for pre-processing")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug_num", type=int, default=2000)
    parser.add_argument("--sorted", action="store_true",
                        help='Sort the data so most batches have the same input length,'
                             ' makes things about 2x faster. Our experiments did not actually'
                             ' use this in the end (not sure if it makes a difference) so '
                             'its off by default.')
    parser.add_argument("--which_bias", choices=["hans", "hypo", "hans_json", "mix", "dam",'qqp_hans_json','fever_claim_only_balanced'], required=True)
    parser.add_argument("--custom_teacher", default=None)
    parser.add_argument("--custom_bias", default=None)
    parser.add_argument("--theta", type=float, default=0.1, help="for theta smoothed distillation loss")
    parser.add_argument("--add_bias_on_eval", action="store_true")

    args = parser.parse_args()

    utils.add_stdout_logger()

    if args.mode == "none":
        loss_fn = clf_distill_loss_functions.Plain()
    elif args.mode == "distill":
        loss_fn = clf_distill_loss_functions.DistillLoss()
    elif args.mode == "smoothed_distill":
        loss_fn = clf_distill_loss_functions.SmoothedDistillLoss()
    elif args.mode == "smoothed_distill_annealed":
        loss_fn = clf_distill_loss_functions.SmoothedDistillLossAnnealed()
    elif args.mode == "theta_smoothed_distill":
        loss_fn = clf_distill_loss_functions.ThetaSmoothedDistillLoss(args.theta)
    elif args.mode == "label_smoothing":
        loss_fn = clf_distill_loss_functions.LabelSmoothing(3)
    elif args.mode == "reweight_baseline":
        loss_fn = clf_distill_loss_functions.ReweightBaseline()
    elif args.mode == "permute_smoothed_distill":
        loss_fn = clf_distill_loss_functions.PermuteSmoothedDistillLoss()
    elif args.mode == "smoothed_reweight_baseline":
        loss_fn = clf_distill_loss_functions.SmoothedReweightLoss()
    elif args.mode == "bias_product_baseline":
        loss_fn = clf_distill_loss_functions.BiasProductBaseline()
    elif args.mode == "learned_mixin_baseline":
        loss_fn = clf_distill_loss_functions.LearnedMixinBaseline(args.penalty)
    elif args.mode == "reweight_by_teacher":
        loss_fn = clf_distill_loss_functions.ReweightByTeacher()
    elif args.mode == "reweight_by_teacher_annealed":
        loss_fn = clf_distill_loss_functions.ReweightByTeacherAnnealed()
    elif args.mode == "bias_product_by_teacher":
        loss_fn = clf_distill_loss_functions.BiasProductByTeacher()
    elif args.mode == "bias_product_by_teacher_annealed":
        loss_fn = clf_distill_loss_functions.BiasProductByTeacherAnnealed()
    elif args.mode == "focal_loss":
        loss_fn = clf_distill_loss_functions.FocalLoss(gamma=args.focal_loss_gamma)
    else:
        raise RuntimeError()

    output_dir = args.output_dir

    if args.do_train:
        if exists(output_dir):
            if len(os.listdir(output_dir)) > 0:
                logging.warning("Output dir exists and is non-empty")
        else:
            os.makedirs(output_dir)

    print("Saving model to %s" % output_dir)

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logging.info(
        "device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
            device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError(
            "Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(222)
        if n_gpu > 0:
            torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval and not args.do_test:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if os.path.exists(output_dir) and os.listdir(output_dir) and args.do_train:
        logging.warning(
            "Output directory ({}) already exists and is not empty.".format(output_dir))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Its way ot easy to forget if this is being set by a command line flag
    if "-uncased" in args.bert_model:
        do_lower_case = True
    elif "-cased" in args.bert_model:
        do_lower_case = False
    else:
        raise NotImplementedError(args.bert_model)

    tokenizer = BertTokenizer.from_pretrained(args.bert_model,
                                              do_lower_case=do_lower_case)

    num_train_optimization_steps = None
    train_examples = None
    if args.do_train:
      	# todo 根据任务选择
        if args.task == 'mnli':
            train_examples = load_mnli(True, args.debug_num if args.debug else None)
            logging.info("***** gonna handle mnli task *****")
        elif args.task == 'qqp':
            train_examples = load_qqp(True, args.debug_num if args.debug else None)
            logging.info("***** gonna handle qqp task *****")
        else:
            train_examples = load_fever(True, args.debug_num if args.debug else None)
            logging.info("***** gonna handle fever task *****")

        num_train_optimization_steps = int(
            len(
                train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
        if args.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()
        loss_fn.num_train_optimization_steps = int(num_train_optimization_steps)
        loss_fn.num_epochs = int(args.num_train_epochs)

    # Prepare model
    cache_dir = args.cache_dir if args.cache_dir else os.path.join(
        str(PYTORCH_PRETRAINED_BERT_CACHE),
        'distributed_{}'.format(args.local_rank))

    # todo num_labels
    if args.task == 'qqp':
        label_num = 2
    else:
        label_num = 3
    model = BertDistill.from_pretrained(
        args.bert_model, cache_dir=cache_dir, num_labels=label_num, loss_fn=loss_fn)
    # todo num_labels

    if args.fp16:
        model.half()
    model.to(device)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]
    if args.fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              bias_correction=False,
                              max_grad_norm=1.0)
        if args.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)

    else:
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=num_train_optimization_steps)

    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0

    if args.do_train:
        train_features: List[InputFeatures] = convert_examples_to_features(
            train_examples, args.max_seq_length, tokenizer, args.n_processes)

        if args.which_bias == "mix":
            hypo_bias_map = load_bias("hypo")
            hans_bias_map = load_bias("hans")
            bias_map = {}
            def compute_entropy(probs, base=label_num):
                return -(probs * (np.log(probs) / np.log(base))).sum()
            for key in hypo_bias_map.keys():
                hypo_ent = compute_entropy(np.exp(hypo_bias_map[key]))
                hans_ent = compute_entropy(np.exp(hans_bias_map[key]))
                if hypo_ent < hans_ent:
                    bias_map[key] = hypo_bias_map[key]
                else:
                    bias_map[key] = hans_bias_map[key]
        else:
            bias_map = load_bias(args.which_bias, custom_path=args.custom_bias)

        for fe in train_features:
            fe.bias = bias_map[fe.example_id].astype(np.float32)
        teacher_probs_map = load_teacher_probs(args.custom_teacher)
        for fe in train_features:
            fe.teacher_probs = np.array(teacher_probs_map[fe.example_id]).astype(
                np.float32)

        example_map = {}
        for ex in train_examples:
            example_map[ex.id] = ex

        logging.info("***** Running training *****")
        logging.info("  Num examples = %d", len(train_examples))
        logging.info("  Batch size = %d", args.train_batch_size)
        logging.info("  Num steps = %d", num_train_optimization_steps)

        train_dataloader = build_train_dataloader(train_features, args.train_batch_size,
                                                  args.seed, args.sorted)

        model.train()
        loss_ema = 0
        total_steps = 0
        decay = 0.99

        for _ in trange(int(args.num_train_epochs), desc="Epoch", ncols=100):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            pbar = tqdm(train_dataloader, desc="loss", ncols=100)
            for step, batch in enumerate(pbar):
                batch = tuple(t.to(device) for t in batch)
                if bias_map is not None:
                    example_ids, input_ids, input_mask, segment_ids, label_ids, bias, teacher_probs = batch
                else:
                    bias = None
                    example_ids, input_ids, input_mask, segment_ids, label_ids = batch

                logits, loss = model(input_ids, segment_ids, input_mask, label_ids, bias,
                                     teacher_probs)

                total_steps += 1
                loss_ema = loss_ema * decay + loss.cpu().detach().numpy() * (1 - decay)
                descript = "loss=%.4f" % (loss_ema / (1 - decay ** total_steps))
                pbar.set_description(descript, refresh=False)

                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        # modify learning rate with special warm up BERT uses
                        # if args.fp16 is False, BertAdam is used that handles this automatically
                        lr_this_step = args.learning_rate * warmup_linear(
                            global_step / num_train_optimization_steps,
                            args.warmup_proportion)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

        # Save a trained model and the associated configuration
        model_to_save = model.module if hasattr(model,
                                                'module') else model  # Only save the model it-self
        output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
        torch.save(model_to_save.state_dict(), output_model_file)
        output_config_file = os.path.join(output_dir, CONFIG_NAME)
        with open(output_config_file, 'w') as f:
            f.write(model_to_save.config.to_json_string())

        # Record the args as well
        arg_dict = {}
        for arg in vars(args):
            arg_dict[arg] = getattr(args, arg)
        with open(join(output_dir, "args.json"), 'w') as out_fh:
            json.dump(arg_dict, out_fh)

        # Load a trained model and config that you have fine-tuned
        config = BertConfig(output_config_file)
	# todo 将num_labels变量的值改为label_num
        model = BertDistill(config, num_labels=label_num, loss_fn=loss_fn)
        model.load_state_dict(torch.load(output_model_file))
    else:
        output_config_file = os.path.join(output_dir, CONFIG_NAME)
        config = BertConfig.from_json_file(output_config_file)
        output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
        model = BertDistill(config, num_labels=label_num, loss_fn=loss_fn)
        model.load_state_dict(torch.load(output_model_file))

    model.to(device)

    if not args.do_eval and not args.do_test:
        return
    if not (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        return

    model.eval()

    if args.do_eval:
      
        # todo 根据特定任务进行了修改
        if args.task == 'mnli':
            eval_datasets = [("mnli_dev_m", load_mnli(False)),
                             ("mnli_dev_mm", load_mnli(False, custom_path="dev_mismatched.tsv"))]
            # eval_datasets += load_easy_hard(prefix="overlap_", no_mismatched=True)
            # eval_datasets += load_easy_hard()
            eval_datasets += [("hans", load_hans())]
            eval_datasets += load_hans_subsets()
        elif args.task == 'fever':
            eval_datasets = [("fever_dev",load_fever(False)),("symmetric",load_Symmetric()),("symmetric_test",load_Symmetric_test()),("symmetric_full",load_Symmetric_full()) ]
        else: # qqp
            eval_datasets = [("qqp_dev",load_qqp(False)),("paws",load_paws(is_train=True)),("paws_test",load_paws(is_train=False))]

        # todo 根据特定任务进行了修改
        # stress test
        # eval_datasets += [("negation_m", load_jsonl("multinli_0.9_negation_matched.jsonl",
        #                                             "../dataset/StressTests/Negation"))]
        # eval_datasets += [("negation_mm", load_jsonl("multinli_0.9_negation_mismatched.jsonl",
        #                                              "../dataset/StressTests/Negation"))]
        # eval_datasets += [("overlap_m", load_jsonl("multinli_0.9_taut2_matched.jsonl",
        #                                             "../dataset/StressTests/Word_Overlap"))]
        # eval_datasets += [("overlap_mm", load_jsonl("multinli_0.9_taut2_mismatched.jsonl",
        #                                              "../dataset/StressTests/Word_Overlap"))]
        # eval_datasets += [("length_m", load_jsonl("multinli_0.9_length_mismatch_matched.jsonl",
        #                                             "../dataset/StressTests/Length_Mismatch"))]
        # eval_datasets += [("length_mm", load_jsonl("multinli_0.9_length_mismatch_mismatched.jsonl",
        #                                              "../dataset/StressTests/Length_Mismatch"))]

        # eval_datasets = [("rte", load_jsonl("eval_rte.jsonl",
        #                                     "../dataset/mnli_eval_suite"))]
        # eval_datasets += [("rte_glue", load_jsonl("eval_glue_rte.jsonl",
        #                                          "../dataset/mnli_eval_suite"))]
        # eval_datasets += [("sick", load_jsonl("eval_sick.jsonl",
        #                                       "../dataset/mnli_eval_suite"))]
        # eval_datasets += [("diagnostic", load_jsonl("diagnostic-full.jsonl",
        #                                             "../dataset/mnli_eval_suite"))]
        # eval_datasets += [("scitail", load_jsonl("scitail_1.0_test.txt",

        #                                           "../dataset/scitail/snli_format"))]

        # todo delete
        if args.do_eval_on_train:
           
	    # todo 根据任务修改
            if args.task == 'mnli':
                eval_datasets = [("mnli_train", load_mnli(True))]
            elif args.task == 'fever':
                eval_datasets = [("fever_train", load_fever(True))]
            else: # qqp
                eval_datasets = [("qqp_train", load_qqp(True))]
    else:
        eval_datasets = []

        # todo 加载需要得到数shallow特征的数据集
    if args.get_bert_output:
        if args.task == 'mnli':
            if args.do_eval_on_train:
                eval_datasets = [("mnli", load_mnli(is_train=True))]
            else:
                if args.gene_challenge:  # 加载挑战集
                    eval_datasets = [("hans", load_hans())]
                else:  # 加载dev集合
                    eval_datasets = [("mnli_dev", load_mnli(is_train=False))]
        elif args.task == 'fever':
            if args.do_eval_on_train:
                eval_datasets = [("fever", load_fever(is_train=True))]
            else:
                if args.gene_challenge:  # 加载挑战集
                    eval_datasets = [("symmetric_full", load_Symmetric_full())]
                else:  # 加载dev集合
                    eval_datasets = [("fever_dev", load_fever(is_train=False))]
        else: # qqp
            if args.do_eval_on_train:
                eval_datasets = [("qqp", load_qqp(is_train=True))]
            else:
                if args.gene_challenge:  # 加载挑战集
                    eval_datasets = [("paws", load_paws(is_train=True))]
                else:  # 加载dev集合
                    eval_datasets = [("qqp_dev", load_qqp(False))]

    if args.do_test:
        test_datasets = load_all_test_jsonl()
        eval_datasets += test_datasets
        subm_paths = ["../submission/{}.csv".format(x[0]) for x in test_datasets]

    for ix, (name, eval_examples) in enumerate(eval_datasets):
        logging.info("***** Running evaluation on %s *****" % name)
        logging.info("  Num examples = %d", len(eval_examples))
        logging.info("  Batch size = %d", args.eval_batch_size)


        # todo final
        shallow_features = pickle.load(open(args.shallow_feature_file, 'rb'))
        eval_features = convert_examples_to_features(
            eval_examples, args.max_seq_length, tokenizer,shallow_features=shallow_features)
        pickle.dump(eval_features,open(args.final_feature,'wb'))
        logging.info("final features have been dumped to " + args.final_feature)
        return
        # todo


        eval_features.sort(key=lambda x: len(x.input_ids))
        all_label_ids = np.array([x.label_id for x in eval_features])
        eval_dataloader = build_eval_dataloader(eval_features, args.eval_batch_size)

        eval_loss = 0
        nb_eval_steps = 0
        probs = []
        test_subm_ids = [
]
        # todo 新增list用于存储bert输出 现在要改成新建或读取已存在的词典，因为id要与其一一对应
        output_list = {}
        if exists(args.shallow_feature_file):
            with open(args.shallow_feature_file,'rb') as shallow_f:
                output_list = pickle.load(shallow_f)  # 读入字典，向后面继续添加一行
        # todo 新增list用于存储bert输出        

        for example_ids, input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader,
                                                                               desc="Evaluating",
                                                                               ncols=100):
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            label_ids = label_ids.to(device)

            with torch.no_grad():
                logits = model(input_ids, segment_ids, input_mask)

                # todo 可以输出bert后的特征向量表示，转为cpu形式后存储在list中，随后用pickle保存到文件里
                if args.get_bert_output:  # 如果需要bert的CLS输出
                    pooled_output = model.get_bert_output(input_ids, segment_ids, input_mask).data
                    pooled_output = pooled_output.cpu()

                    # todo
                    exm_id = example_ids.cpu().numpy()
                    if not exm_id in output_list:
                        output_list[exm_id] = []
                    else:
                        output_list[exm_id].append(pooled_output)
                    # todo

                    # output_list.append(pooled_output)  # 加入list
                    # print("bert中得到的特征表示：")
                    # print(pooled_output)
                # todo 可以输出bert后的特征向量表示，转为cpu形式后存储在list中，随后用pickle保存到文件里


            # create eval loss and other metric required by the task
            loss_fct = CrossEntropyLoss()
           
	    
            if args.task == 'qqp':
                tmp_eval_loss = loss_fct(logits.view(-1, 2), label_ids.view(-1))
            else:
                tmp_eval_loss = loss_fct(logits.view(-1, 3), label_ids.view(-1))

            eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            probs.append(torch.nn.functional.softmax(logits, 1).detach().cpu().numpy())
            test_subm_ids.append(example_ids.cpu().numpy())

        # todo 将list写入到文件中
        logging.info("********************************")
        logging.info(len(output_list[list(output_list.keys())[0]]))  # 取一个样本 提示输出现在有多少行了
        logging.info("********************************")
        pickle.dump(output_list,open(args.shallow_feature_file,"wb"))
        # todo

        probs = np.concatenate(probs, 0)
        test_subm_ids = np.concatenate(test_subm_ids, 0)
        eval_loss = eval_loss / nb_eval_steps

        if "hans" in name:
            # take max of non-entailment rather than taking their sum
            probs[:, 0] = probs[:, [0, 2]].max(axis=1)
            # probs[:, 0] = probs[:, 0] + probs[:, 2]
            probs = probs[:, :2]

        preds = np.argmax(probs, axis=1)

        
    	# todo QQP需要的评测标准F1
        if args.task == 'qqp':
            result = acc_and_f1(preds, all_label_ids)
        elif args.task == 'mnli':
            result = {"acc": simple_accuracy(preds, all_label_ids)}
            result["loss"] = eval_loss

            conf_plot_file = os.path.join(output_dir, "eval_%s_confidence.png" % name)
            ECE, bins_acc, bins_conf, bins_num = visualize_predictions(probs, all_label_ids,
                                                                       conf_plot_file=conf_plot_file)
            result["ECE"] = ECE
            result["bins_acc"] = bins_acc
            result["bins_conf"] = bins_conf
            result["bins_num"] = bins_num
        elif args.task == 'fever':
            result = {"acc": simple_accuracy(preds, all_label_ids)}
        # todo QQP需要的评测标准F1

        output_eval_file = os.path.join(output_dir, "eval_%s_results.txt" % name)
        output_all_eval_file = os.path.join(output_dir, "eval_all_results.txt")
        with open(output_eval_file, "w") as writer, open(output_all_eval_file, "a") as all_writer:
            logging.info("***** Eval results *****")
            all_writer.write("eval results on %s:\n" % name)
            for key in sorted(result.keys()):
                logging.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))
                all_writer.write("%s = %s\n" % (key, str(result[key])))

        output_answer_file = os.path.join(output_dir, "eval_%s_answers.json" % name)
        answers = {ex.example_id: [float(x) for x in p] for ex, p in
                   zip(eval_features, probs)}
        with open(output_answer_file, "w") as f:
            json.dump(answers, f)

        # prepare submission file
        if args.do_test and ix >= len(eval_datasets) - len(test_datasets):
            with open(subm_paths.pop(0), "w") as subm_f:
                subm_f.write("pairID,gold_label\n")
                for sub_id, pred_label_id in zip(test_subm_ids, preds):
                    subm_f.write("{},{}\n".format(str(sub_id), REV_NLI_LABEL_MAP[pred_label_id]))


if __name__ == "__main__":
    main()


# eval_datasets = [("snli_test", load_jsonl(file_path="snli_1.0_test.jsonl", data_dir="../dataset/snli")),
#                  ("snli_hard_test", load_jsonl(file_path="snli_1.0_test_hard.jsonl", data_dir="../dataset/snli")),
#                  ("snli_lovp_e", load_jsonl(file_path="snli_overlap_entailment.jsonl", data_dir="../dataset/snli")),
#                  ("snli_lovp_ne", load_jsonl(file_path="snli_overlap_non_entailment.jsonl", data_dir="../dataset/snli"))]
