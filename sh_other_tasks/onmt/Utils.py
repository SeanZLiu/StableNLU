import argparse
import json
import logging
import os
import random
from collections import namedtuple
from os.path import join, exists
from typing import List, Dict, Iterable
import collections

# temporary hack for the pythonroot issue
import sys

import numpy as np
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.modeling import BertConfig, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.modeling import BertForSequenceClassification
from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.tokenization import BertTokenizer
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Dataset, \
    Sampler
from tqdm import trange, tqdm

import onmt.clf_distill_loss_functions
from onmt.bert_distill import BertDistill
from onmt.clf_distill_loss_functions import *

import pickle
import pandas as pd
from os import makedirs
from os.path import dirname
from typing import TypeVar
from sklearn.cluster import SpectralClustering
from sklearn import metrics

from multiprocessing import Lock
from multiprocessing import Pool
from typing import Iterable, List
from onmt import config

import requests
import copy
import pdb


T = TypeVar('T')

NLI_LABELS = ["contradiction", "entailment", "neutral"]
NLI_LABEL_MAP = {k: i for i, k in enumerate(NLI_LABELS)}
REV_NLI_LABEL_MAP = {i: k for i, k in enumerate(NLI_LABELS)}
NLI_LABEL_MAP["hidden"] = NLI_LABEL_MAP["entailment"]

TextPairExample = namedtuple("TextPairExample", ["id", "premise", "hypothesis", "label"])


def add_stdout_logger():
    """Setup stdout logging"""

    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s',
                                  datefmt='%m/%d/%Y %H:%M:%S', )
    handler.setFormatter(formatter)
    handler.setLevel(logging.INFO)

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(handler)


def ensure_dir_exists(filename):
    """Make sure the parent directory of `filename` exists"""
    makedirs(dirname(filename), exist_ok=True)


def download_to_file(url, output_file):
    """Download `url` to `output_file`, intended for small files."""
    ensure_dir_exists(output_file)
    with requests.get(url) as r:
        r.raise_for_status()
        with open(output_file, 'wb') as f:
            f.write(r.content)


def load_pickle(filename):
    """Load an object from a pickled file."""
    with open(filename, "rb") as f:
        return pickle.load(f)


# ++++++++++++++++++++++++++++++++++++++++++++++++++
# data processing functions

def flatten_list(iterable_of_lists: Iterable[Iterable[T]]) -> List[T]:
    """Unpack lists into a single list."""
    return [x for sublist in iterable_of_lists for x in sublist]


def split(lst: List[T], n_groups) -> List[List[T]]:
    """ partition `lst` into `n_groups` that are as evenly sized as possible  """
    per_group = len(lst) // n_groups
    remainder = len(lst) % n_groups
    groups = []
    ix = 0
    for _ in range(n_groups):
        group_size = per_group
        if remainder > 0:
            remainder -= 1
            group_size += 1
        groups.append(lst[ix:ix + group_size])
        ix += group_size
    return groups


def group(lst: List[T], max_group_size) -> List[List[T]]:
    """partition `lst` into that the mininal number of groups that as evenly sized
    as possible  and are at most `max_group_size` in size """
    if max_group_size is None:
        return [lst]
    n_groups = (len(lst) + max_group_size - 1) // max_group_size
    per_group = len(lst) // n_groups
    remainder = len(lst) % n_groups
    groups = []
    ix = 0
    for _ in range(n_groups):
        group_size = per_group
        if remainder > 0:
            remainder -= 1
            group_size += 1
        groups.append(lst[ix:ix + group_size])
        ix += group_size
    return groups


class Processor:

    def process(self, data: Iterable):
        """Map elements to an unspecified output type, the output but type must None or
        be able to be aggregated with the  `+` operator"""
        raise NotImplementedError()

    def finalize_chunk(self, data):
        """Finalize the output from `preprocess`, in multi-processing senarios this will still be run on
         the main thread so it can be used for things like interning"""
        pass


def _process_and_count(questions: List, preprocessor: Processor):
    count = len(questions)
    output = preprocessor.process(questions)
    return output, count


def process_par(data: List, processor: Processor, n_processes,
                chunk_size=1000, desc=None, initializer=None):
    """Runs `processor` on the elements in `data`, possibly in parallel, and monitor with tqdm"""

    if chunk_size <= 0:
        raise ValueError("Chunk size must be >= 0, but got %s" % chunk_size)
    if n_processes is not None and n_processes <= 0:
        raise ValueError("n_processes must be >= 1 or None, but got %s" % n_processes)
    n_processes = min(len(data), 1 if n_processes is None else n_processes)

    if n_processes == 1 and not initializer:
        out = processor.process(tqdm(data, desc=desc, ncols=80))
        processor.finalize_chunk(out)
        return out
    else:
        chunks = split(data, n_processes)
        chunks = flatten_list([group(c, chunk_size) for c in chunks])
        total = len(data)
        pbar = tqdm(total=total, desc=desc, ncols=80)
        lock = Lock()

        def call_back(results):
            processor.finalize_chunk(results[0])
            with lock:
                pbar.update(results[1])

        with Pool(n_processes, initializer=initializer) as pool:
            results = [
                pool.apply_async(_process_and_count, [c, processor], callback=call_back)
                for c in chunks
            ]
            results = [r.get()[0] for r in results]

        pbar.close()
        output = results[0]
        if output is not None:
            for r in results[1:]:
                output += r
        return output


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
        download_to_file(HANS_URL, src)

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


def load_hans(n_samples=None, filter_label=None, filter_subset=None) -> List[TextPairExample]:
    out = []

    if filter_label is not None and filter_subset is not None:
        logging.info("Loading hans subset: {}-{}...".format(filter_label, filter_subset))
    else:
        logging.info("Loading hans all...")

    src = join(config.HANS_SOURCE, "heuristics_evaluation_set.txt")
    if not exists(src):
        logging.info("Downloading source to %s..." % config.HANS_SOURCE)
        download_to_file(HANS_URL, src)

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
        #file_path = config.TEACHER_SOURCE
        file_path = "../teacher_preds/mnli_trained_on_sample2K_seed111_self.json"
    else:
        #file_path = custom_teacher
        file_path = "../teacher_preds/mnli_trained_on_sample2K_seed111_self.json"
    
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
        bias = load_pickle(bias_src)
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

    def __init__(self, example_id, input_ids, segment_ids, label_id, bias):
        self.example_id = example_id
        self.input_ids = input_ids
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.bias = bias


class ExampleConverter(Processor):
    def __init__(self, max_seq_length, tokenizer):
        self.max_seq_length = max_seq_length
        self.tokenizer = tokenizer

    def process(self, data: Iterable):
        features = []
        tokenizer = self.tokenizer
        max_seq_length = self.max_seq_length

        for example in data:
            tokens_a = tokenizer.tokenize(example.premise)

            tokens_b = None
            if example.hypothesis:
                tokens_b = tokenizer.tokenize(example.hypothesis)
                # Modifies `tokens_a` and `tokens_b` in place so that the total
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

            features.append(
                InputFeatures(
                    example_id=example.id,
                    input_ids=np.array(input_ids),
                    segment_ids=np.array(segment_ids),
                    label_id=example.label,
                    bias=None
                ))
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
    if hasattr(batch[0], 'shallow_features') or hasattr(batch[0], 'bias_features'):
        has_shallow_feature = True
        shallow_features = []
    else:
        has_shallow_feature = False

    if hasattr(batch[0], 'bias_probs'):
        has_bias_probs = True
        bias_probs = []
    else:
        has_bias_probs = False

    if hasattr(batch[0], 'sample_weight'):
        has_sample_weight = True
        sample_weights = []
    else:
        has_sample_weight = False

    if hasattr(batch[0], 'counter_prob'):
        has_counter_res = True
        counter_feature = []
        counter_prob = []
    else:
        has_counter_res = False
       
    for i, ex in enumerate(batch):
        input_ids[i, :len(ex.input_ids)] = ex.input_ids
        segment_ids[i, :len(ex.segment_ids)] = ex.segment_ids
        mask[i, :len(ex.input_ids)] = 1
        if has_shallow_feature:
            try:
                shallow_features_tmp = np.concatenate(ex.shallow_features)
            except:
                shallow_features_tmp = ex.bias_features
            shallow_features_tmp = torch.FloatTensor(shallow_features_tmp).unsqueeze(0)
            shallow_features.append(shallow_features_tmp)

        if has_bias_probs:
            bias_probs_tmp = np.concatenate(ex.bias_probs)
            bias_probs_tmp = torch.FloatTensor(bias_probs_tmp).unsqueeze(0)
            bias_probs.append(bias_probs_tmp)

        if has_sample_weight:
            sample_weight_tmp = torch.FloatTensor([ex.sample_weight]).unsqueeze(0)
            sample_weights.append(sample_weight_tmp)
        
        if has_counter_res:
            counter_prob_tmp = torch.FloatTensor(ex.counter_prob).unsqueeze(0)
            counter_feature_tmp =  torch.FloatTensor(ex.counter_feature)

            counter_prob.append(counter_prob_tmp)
            counter_feature.append(counter_feature_tmp)

    input_ids = torch.as_tensor(input_ids)
    segment_ids = torch.as_tensor(segment_ids)
    label_ids = torch.as_tensor(np.array([x.label_id for x in batch], np.int64))
    if has_shallow_feature:
        shallow_features = torch.cat(shallow_features)

    if has_bias_probs:
        bias_probs = torch.cat(bias_probs)

    if has_sample_weight:
        sample_weights = torch.cat(sample_weights)

    if has_counter_res:
        counter_prob = torch.cat(counter_prob)
        counter_feature = torch.cat(counter_feature)

    # include example ids for test submission
    try:
        example_ids = torch.tensor([int(x.example_id) for x in batch])
    except:
        example_ids = torch.zeros(len(batch)).long()
        
    return_ls = [example_ids, input_ids, mask, segment_ids, label_ids]
    
    if has_shallow_feature:
        return_ls.append(shallow_features)
    if has_sample_weight:
        return_ls.append(sample_weights)
    if has_bias_probs:
        return_ls.append(bias_probs)
    if has_counter_res:
        return_ls.append(counter_feature)
        return_ls.append(counter_prob)
    return return_ls

'''
if batch[0].bias is None and has_shallow_feature and (not has_sample_weight):
    return example_ids, input_ids, mask, segment_ids, label_ids, shallow_features
elif has_sample_weight and has_shallow_feature:
    return example_ids, input_ids, mask, segment_ids, label_ids, shallow_features, sample_weights
elif has_sample_weight and (not has_shallow_feature): 
    return example_ids, input_ids, mask, segment_ids, label_ids, sample_weights
else:
    return example_ids, input_ids, mask, segment_ids, label_ids

teacher_probs = torch.tensor([x.teacher_probs for x in batch])
bias = torch.tensor([x.bias for x in batch])

return example_ids, input_ids, mask, segment_ids, label_ids, bias, teacher_probs
'''

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


def convert_examples_to_features(
        examples: List[TextPairExample], max_seq_length, tokenizer, n_process=1):
    converter = ExampleConverter(max_seq_length, tokenizer)
    return process_par(examples, converter, n_process, chunk_size=2000, desc="featurize")


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


def f1(preds, labels):
    f1_ls = []
    for i in [0,1]:
        precision = ((preds == labels) * (preds == i)).sum() / (preds == i).sum()
        recall = ((preds == labels) * (preds == i)).sum() / (labels == i).sum()
        f1_tmp = 2 * precision * recall / (precision + recall)
        f1_ls.append(f1_tmp)

    return f1_ls

'''
def f1(preds, labels):

    #return [simple_accuracy(preds[labels==0], labels[labels==0]),simple_accuracy(preds[labels==1], labels[labels==1])]
    return simple_accuracy(preds, labels)
'''

def fetch_loss_fuction(args):    
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
        
    return loss_fn

def do_evaluation(model, data_loader, gpu_id, name='none', use_bias_feature=False, mode='mean', weight=1, disturb=False, measure='acc'):
    
    nb_eval_steps = 0
    all_label_ids = []
    probs = []
    pooled_outs = []
    test_subm_ids = []
    all_example_ids = []
    
    pbar = tqdm(data_loader,desc="Evaluating",ncols=100)
    
    for batch in pbar:
        if len(batch) == 5:
            example_ids, input_ids, input_mask, segment_ids, label_ids = batch
        elif len(batch) == 6:
            example_ids, input_ids, input_mask, segment_ids, label_ids, bias_features = batch
        elif len(batch) == 7:
            example_ids, input_ids, input_mask, segment_ids, label_ids, bias_features, sample_weights = batch
        elif len(batch) == 8:
            example_ids, input_ids, input_mask, segment_ids, label_ids, bias_features, sample_weights, _ = batch
        try:
            input_ids = input_ids.to(gpu_id)
            input_mask = input_mask.to(gpu_id)
            segment_ids = segment_ids.to(gpu_id)
            label_ids = label_ids.to(gpu_id)
        except:
            pdb.set_trace()

        if use_bias_feature:
            bias_features = bias_features.to(gpu_id)
        
        with torch.no_grad():          
            logits, pooled_out = model(input_ids, segment_ids, input_mask)
        '''    
        if disturb:
            param_optimizer = list(model.named_parameters())
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            
            optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': opt.l2_reg},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
            ]    
            optimizer = BertAdam(optimizer_grouped_parameters,lr = 5e-5,warmup=0.0,t_total=20000)   
            
            loss = torch.nn.CrossEntropyLoss(logits, label_ids)            
            loss.backward()
            
            logits, pooled_out = model(input_ids, segment_ids, input_mask, disturb=True)
            
            model.zero_grad()
        '''    
        if use_bias_feature:
            if mode == 'mean':
                bias_feature = bias_features.mean(dim=1) * weight
            elif mode == 'random':
                rand_id = random.randint(0,4) * weight
                bias_feature = bias_features[:,rand_id, :]
            elif mode == 'reweight':
                bias_feature = bias_features[:,0, :] * weight
            elif mode == 'mean_aux':
                bias_feature = torch.tanh(model.module.aux_linear(bias_features.mean(dim=1))) 
            elif mode == 'reweight_dy':
                bias_feature = bias_features.mean(dim=1) 
            elif mode == 'reweight_dy_inv':
                bias_feature = bias_features.mean(dim=1) 
            else:
                bias_feature = bias_features.mean(dim=1) * weight
  
            pooled_out = torch.tanh(pooled_out + bias_feature)
            if hasattr(model, 'module'):
                logits = model.module.classifier(pooled_out)
            else:
                logits = model.classifier(pooled_out)
        # create eval loss and other metric required by the task
        nb_eval_steps += 1
        all_label_ids.append(label_ids.cpu().numpy())
        all_example_ids.append(example_ids)
        probs.append(torch.nn.functional.softmax(logits, 1).detach().cpu().numpy())
        pooled_outs.append(pooled_out.detach().cpu().numpy())        
        test_subm_ids.append(example_ids.cpu().numpy())
    
    all_example_ids = np.concatenate(all_example_ids, 0)
    all_label_ids = np.concatenate(all_label_ids, 0)
    probs = np.concatenate(probs, 0)
    pooled_outs = np.concatenate(pooled_outs, 0)
    test_subm_ids = np.concatenate(test_subm_ids, 0)
    
    if "hans" in name:
        # take max of non-entailment rather than taking their sum
        probs[:, 0] = probs[:, [0, 2]].max(axis=1)
        # probs[:, 0] = probs[:, 0] + probs[:, 2]
        probs = probs[:, :2]

    preds = np.argmax(probs, axis=1)
    
    if max(all_label_ids) == 1:
         preds = preds[238:]
         all_label_ids = all_label_ids[238:]
    
    # pdb.set_trace()    
    if measure == 'acc':
        result = {"acc": simple_accuracy(preds, all_label_ids), 'probs':probs, 'pooled_outs':pooled_outs, 'labels':all_label_ids, 'example_ids': all_example_ids}
    elif measure == 'f1':
        result = {"acc": f1(preds, all_label_ids), 'probs':probs, 'pooled_outs':pooled_outs, 'labels':all_label_ids, 'example_ids': all_example_ids}
    return result
    
    
def jaccard(x,y):
    x = set(x)
    y = set(y)
    score = len(x.intersection(y)) / len(x.union(y))
    return score
    

def confused_pair_gen(res_A, res_B, args, train_features=None, features_exam=None):
                      
    ids_A = res_A['example_ids']
    ids_B = res_B['example_ids']

    label_A = res_A['labels']
    label_B = res_B['labels']
    
    prob_A = res_A['probs']
    prob_B = res_B['probs']
    
    f_features_A = res_A['pooled_outs']
    f_features_B = res_B['pooled_outs']
    
    candi_A_ls = np.where((prob_A.argmax(1) == label_A) * (np.array(prob_A.max(1)) > args.thre_posi) )[0]
    candi_B_ls = np.where((prob_B.argmax(1) != label_B) * (np.array(prob_B.max(1)) > args.thre_nega) )[0]

    # candi_C_ls = np.where((prob_A.argmax(1) != label_A) * (np.array(prob_A.max(1)) > 0.95) )[0]
    # candi_D_ls = np.where((prob_B.argmax(1) == label_B) * (np.array(prob_B.max(1)) > 0.95) )[0]

    # candi_A_ls = np.concatenate([candi_A_ls, candi_C_ls])
    # candi_B_ls = np.concatenate([candi_B_ls, candi_D_ls])

    #candi_A_ls = np.where((prob_A.argmax(1) == label_A) * (np.array(prob_A.max(1)) > args.thre_posi) * (prob_A.argmax(1) == prob_B.argmax(1)))[0]
    #candi_B_ls = np.where((prob_B.argmax(1) != label_B) * (np.array(prob_B.max(1)) > args.thre_nega) * (prob_A.argmax(1) == prob_B.argmax(1)))[0]
    
    split_size = 10000
    
    num_split_A = len(candi_A_ls) // split_size
    num_split_B = len(candi_B_ls) // split_size
    
    posi_ls = []
    nega_ls = []
    inp_ls = []
    cum_sample_A = 0
    
    if (len(candi_A_ls) > 0) and (len(candi_B_ls) > 0):
        for ith_split in range(min(num_split_A + 1, 10)):
            cum_sample_B = 0
            for jth_split in range(min(num_split_B + 1, 10)):
                print(ith_split, jth_split)
                try:
                    f_features_A_tmp = f_features_A[candi_A_ls[(ith_split * split_size) : ((ith_split + 1) * split_size)]]
                    f_features_B_tmp = f_features_B[candi_B_ls[(jth_split * split_size) : ((jth_split + 1) * split_size)]]
                    
                    dot_tmp = f_features_A_tmp.dot(f_features_B_tmp.T)
                    try:
                        norm_A_tmp = np.linalg.norm(f_features_A_tmp, axis=-1).reshape(f_features_A_tmp.shape[0], -1)
                        norm_B_tmp = np.linalg.norm(f_features_B_tmp, axis=-1).reshape(-1, f_features_B_tmp.shape[0])
                        
                        norm_tmp = norm_A_tmp.dot(norm_B_tmp)
                        
                        inp = dot_tmp / norm_tmp
                        
                        del dot_tmp, norm_A_tmp, norm_B_tmp, norm_tmp
                        
                        abv_tau_tmp = np.where(inp > args.tau)
                        
                        posi_ls_tmp = (cum_sample_A + abv_tau_tmp[0])
                        nega_ls_tmp = (cum_sample_B + abv_tau_tmp[1])
                        
                        cum_sample_B += f_features_B_tmp.shape[0]
                        inp_ls_tmp = inp[abv_tau_tmp].tolist()
                        
                        posi_ls.extend(candi_A_ls[posi_ls_tmp]) 
                        nega_ls.extend(candi_B_ls[nega_ls_tmp])
    
                        inp_ls.extend(inp_ls_tmp)
                        
                    except:
                        pass
                    
                except:
                    pdb.set_trace()
                    print('!!!')
            cum_sample_A += f_features_A_tmp.shape[0]
    
    del f_features_A, f_features_B
    
    def u(x):
        return x.unique().tolist()
        
    def tolist(x):
        return x.tolist()

    confused_pairs = pd.DataFrame(np.array([posi_ls, nega_ls]).T,columns=['x','y'])
    confused_pairs = confused_pairs.groupby('x')
    confused_pairs = confused_pairs.agg({'x':u, 'y':tolist})
    
    confused_pairs_inv = pd.DataFrame(np.array([posi_ls, nega_ls]).T,columns=['x','y'])
    confused_pairs_inv = confused_pairs_inv.groupby('y')
    confused_pairs_inv = confused_pairs_inv.agg({'x':tolist, 'y':u})
    
    confused_pairs_inv_ls = confused_pairs_inv.iloc[:,0].tolist()
    end_set = set([item for i in confused_pairs_inv.iloc[:,1].tolist() for item in i])
    '''
    dist = np.ones([len(end_set), len(end_set)])
    
    for ith, i in enumerate(confused_pairs_inv_ls):
        for jth, j in enumerate(confused_pairs_inv_ls):
            jac_sim = jaccard(i,j)
            #dist[ith, jth] = dist[ith, jth] - jac_sim
            dist[ith, jth] = dist[ith, jth] - jac_sim
    
    clusters_ls = []
    ch_scores = []
    for i in range(3,10):        
        clusters = SpectralClustering(n_clusters=i,affinity='precomputed').fit_predict(dist)
        ch_score = metrics.calinski_harabasz_score(dist, cluster)
        clusters_ls.append(clusters)
        ch_scores.append(ch_score)
        
    clusters = clusters_ls[np.argmax(ch_scores)]
    
    match_dict = np.ones([prob_B.shape[0], 3])
    for ith, ith_adv_example in enumerate(confused_pairs_inv.iloc[:,1].tolist()):
         match_dict[ith_adv_example, 0] = ith
         match_dict[ith_adv_example, 1] = clusters[ith]
         match_dict[ith_adv_example, 2] = ids_B[ith]
    pdb.set_trace()
    
    def freq_stat(clusters):
        freqs_dict = Counter(clusters)
        freqs = []
        for k in sorted(freqs_dict.keys()):
            freqs.append(freqs_dict[k])
    
        freqs = np.array(freqs)
        return freqs
    
    pdb.set_trace()
        
    freqs_nega = freq_stat(clusters)
    prob_ls = []
    adv_features_ls = []
    
    for i in range(confused_pairs):
        cluster_tmp = freqs_dict[confused_pairs.iloc[i, 1], 1]   
        freq_tmp = freq_stat(cluster_tmp)
        prob_tmp = freq_tmp / freqs_nega
        prob_tmp = prob_tmp / sum(prob_tmp)
        if len(prob_tmp) < len(freqs_nega):
            prob_tmp = np.append(prob_tmp, [0] * (len(freqs_nega) - len(prob_tmp)) )
        prob_ls.append(prob_tmp)
        
        adv_features_tmp = f_features_B[confused_pairs.iloc[i, 1]]
    
    prob_ls = np.array(prob_ls)
    freq_posi = prob_ls.sum(0)
    
    freq_ratio = freq_posi / freqs_nega
    weights = ((1 / freq_ratio) * prob_ls).sum(1) * 3
    '''
    weights_dict = {}
    for i in range(confused_pairs.shape[0]):
        weights_dict[ids_A[confused_pairs.iloc[i, 0]][0]] = 0
    
    for i in end_set:
        weights_dict[ids_B[i]] = 3
        
    max_weight = confused_pairs.iloc[:,1].apply(len).max()
    scale_coef = 2 / max_weight
    
    weights_dict = {}
    counter_examples_dict = {}
    for i in range(confused_pairs.shape[0]):
        weights_dict[ids_A[confused_pairs.iloc[i, 0]][0]] = len(confused_pairs.iloc[i, 1])
        counter_examples_dict[ids_A[confused_pairs.iloc[i, 0]][0]] = confused_pairs.iloc[i, 1]

    if hasattr(train_features[0], 'bias_features'):
        has_bias_feature = True  
    else:     
        has_bias_feature = False
    for ith, train_feature in enumerate(train_features):
        # try:
        #     train_feature.sample_weight = weights_dict[int(train_feature.example_id)] * scale_coef
        # except:
        #     train_feature.sample_weight = 1
        # if ith > 50000:
        #     break

        try:
            adv_features = res_B['pooled_outs'][counter_examples_dict[int(train_feature.example_id)]]
            # pdb.set_trace()
            # adv_features_ts = np.stack(adv_features)
            # adv_features_var = adv_features_ts.var(0) / adv_features_ts.shape[0] + 0.1
            # adv_features_var = adv_features.var(1) + 0.1
            # shallow_fearures_var = train_feature.shallow_features.var(1) / train_feature.shallow_features.shape[0] + 0.1
            # shallow_features_var = np.concatenate(train_feature.shallow_features).var(0) + 0.1
            
            if has_bias_feature:
                train_feature.shallow_features = train_feature.bias_features
            for i in range(min(adv_features.shape[0], len(train_feature.shallow_features))):
                # train_feature.shallow_features[i] = train_feature.shallow_features[i] * 0.1 + adv_features.mean(0) * 0.9
                # train_feature.shallow_features[i] = (train_feature.shallow_features[i] / shallow_features_var + adv_features.mean(0) / adv_features_var) / (1 / adv_features_var + 1 / shallow_features_var)
                if has_bias_feature:
                    train_feature.bias_features[i] = train_feature.bias_features[i] * (1 - args.shallow_feature_weight) + adv_features.mean(0) * args.shallow_feature_weight
                else:
                    train_feature.shallow_features[i] = train_feature.shallow_features[i] * (1 - args.shallow_feature_weight) + adv_features.mean(0) * args.shallow_feature_weight
                # train_feature.shallow_features[i] = - 0.1 * train_feature.shallow_features[i] + adv_features.mean(0) * args.shallow_feature_weight
                
        except KeyError:
            pass

    loader = build_train_dataloader(train_features, args.train_batch_size,args.seed, args.sorted)
                                          
    return confused_pairs, loader
    '''
    confused_samples = []
    
    input_ids_cfs = []
    input_masks_cfs = []
    sentence_inds_cfs = []
    answers_cfs = []
    groupby_cfs = []
    
    for ith, c_samples in enumerate(confused_pairs['y'].tolist()): 
        
        confused_samples.extend(c_samples) 
        
        if not (features_train is None):
        
            c_samples = np.array(c_samples)
            
            input_ids_posi.append(features_train[1][c_samples])
            input_masks_posi.append(features_train[2][c_samples])
            sentence_inds_posi.append(features_train[3][c_samples])
            answers_posi.append(features_train[4][c_samples])
            groupby_cfs.append([ith] * len(c_samples))
                        
    if not features_train is None:
        
        input_ids_posi = torch.cat(input_ids_posi)
        input_masks_posi = torch.cat(input_masks_posi)
        sentence_inds_posi = torch.cat(sentence_inds_posi)
        answers_posi = torch.cat(answers_posi)
        groupby_cfs = torch.cat(groupby_cfs).unsqueeze(-1)
            
        datset_confused_pair = TensorDataset(input_ids_cfs, input_masks_cfs, sentence_inds_cfs, answers_cfs)
        sampler_confused_pair = SequentialSampler(datset_confused_pair)
        datloader_confused_pair = DataLoader(datset_confused_pair, sampler=sampler_confused_pair, batch_size=24)
    
        return confused_pairs, datloader_confused_pair
    else:
        return confused_pairs, None
    '''
'''
def comparison(x, y=None, groupby=None):
    def inp(x, y):
        
        import random
        fold = random.sample([64, 48], 1)[0]
        
        x = x.reshape(-1, fold)
        y = y.reshape(-1, fold)
        
        inp = (((x * y).sum(-1) / x.norm(dim=-1) / y.norm(dim=-1)))
        inp = torch.max(inp, inp * 0).sum()
        #inp = inp.abs() # !!!
        #inp = torch.max(inp, inp * 0 + 0.15).sum() - (inp * 0 + 0.15).sum() # !!!
        return inp

    if groupby is None:
        return inp(x, y)
    else:
        tot_diff = 0
        groupby = groupby.squeeze()
        for l in groupby:
            x_tmp = x[groupby == l]
            x_tmp_mean = x_tmp.mean(dim = -1).unsqueeze(-1)
            x_diff_tmp = x_tmp - x_tmp_mean
            tot_diff += inp(x_diff_tmp, x_diff_tmp)
            
        return tot_diff
'''

def comparision(shallow_probs, counter_probs, labels):
    if len(shallow_probs.shape) > 2:
        shallow_probs = shallow_probs[:,0,:]
    shallow_ind = shallow_probs.argmax(1) == labels
    counter_ind = counter_probs.argmax(1) == labels
    indicator_mat = torch.FloatTensor(np.zeros([shallow_probs.shape[0], 4]))
    indicator_mat = indicator_mat.to(shallow_ind.device)
    for i in range(shallow_probs.shape[0]):
        if shallow_ind[i] and counter_ind[i]:
            indicator_mat[i, 0] = 1
        elif shallow_ind[i] and not counter_ind[i]:
            indicator_mat[i, 1] = 1
        elif not shallow_ind[i] and counter_ind[i]:
            indicator_mat[i, 2] = 1
        else:
            indicator_mat[i, 3] = 1
    return indicator_mat


def datset_split(features, opt):
    L = len(features)
    features_r = features
    
    all_example_ids = torch.tensor([feature.example_id for feature in features_r], dtype=torch.long).unsqueeze(1)
    all_input_ids = torch.tensor(select_field(features_r, 'input_ids'), dtype=torch.long).squeeze()
    all_input_masks = torch.tensor(select_field(features_r, 'input_mask'), dtype=torch.long).squeeze()
    all_sentence_inds = torch.tensor(select_field(features_r, 'sentence_ind'), dtype=torch.long).squeeze()
    all_answers = torch.tensor([f.answer for f in features_r], dtype=torch.long).unsqueeze(1)
    
    all_features = [all_example_ids, all_input_ids, all_input_masks, all_sentence_inds, all_answers]
    
    all_answers_sq = all_answers.squeeze()
        
    features_exam = [feature[-opt.sample_size_e:] for feature in all_features]
    features_train = [feature[:-opt.sample_size_e] for feature in all_features]
   
    def get_data_loader(feature_ls, dat_type='train', batch_size=opt.train_batch_size):
        #test_example_ids, test_input_ids, test_input_masks, test_sentence_inds, test_answers
        data = TensorDataset(feature_ls[0], feature_ls[1], feature_ls[2], feature_ls[3], feature_ls[4])
        
        if dat_type == 'train':
            sampler = RandomSampler(data)
        else:
            sampler = SequentialSampler(data)
        dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)
        
        return dataloader
        
    loader_exam = get_data_loader(features_exam)
    loader_train = get_data_loader(features_train, 'train')
    loader_train_random = get_data_loader(features_train)
    
    return loader_train, loader_train_random, loader_exam


def parse_opt_to_name(opt):
    mode = opt.mode
    dataset = opt.dataset
    Lambda = str(opt.Lambda)
    shallow_weight = opt.shallow_feature_weight
    tau = opt.tau
    tau_annealing = opt.tau_annealing 
    max_candi_list = opt.uplim_candi_list
    sample_size_c = opt.sample_size_c / 10000
    sample_size_e = opt.sample_size_e / 10000
    thre_posi = opt.thre_posi
    thre_nega = opt.thre_nega
    num_train_epochs = opt.num_train_epochs

    name = [str(n) for n in [mode, dataset, Lambda, shallow_weight, tau, tau_annealing, max_candi_list, sample_size_c, sample_size_e, thre_posi, thre_nega, num_train_epochs]]
    name = "_".join(name)
    return name


def ini_from_pretrained(path, de_novo=False, method_combine='add'):
    output_config_file = os.path.join(path, CONFIG_NAME)
    config = BertConfig.from_json_file(output_config_file)
    model = BertDistill(config, num_labels=3, method_combine=method_combine)
    if not de_novo:
        try:
            output_model_file = os.path.join(path, 'stable_e_1pytorch_model.bin')
            model.load_state_dict(torch.load(output_model_file))
        except:
            output_model_file = os.path.join(path, WEIGHTS_NAME)
            model.load_state_dict(torch.load(output_model_file))
    return model
    

def gradient_calc(pred, label, weight_matrix, feature, mode='mean_regu'):
    if not pred.shape == label.shape:
        label_mat = torch.zeros_like(pred)
        for i in label:
            label_mat[i] = 1
        label = label_mat
    d_softmax = (pred - label).unsqueeze(1).transpose(2, 1)
    d_weight_matrix = weight_matrix.expand([d_softmax.shape[0], weight_matrix.shape[0], weight_matrix.shape[1]])
    if mode == 'mean_regu':
        d_feature = (1 - torch.tanh(feature) ** 2).unsqueeze(1)
    # d_feature = feature.unsqueeze(1)
    # 32 * 3 # 768 * 3 # 32 * 768 
    # g = d_softmax * d_weight_matrix * d_feature # 3 * n
    else:
        d_feature = 1
    g = d_softmax * d_weight_matrix * d_feature
    return g
