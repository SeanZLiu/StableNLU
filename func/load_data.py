
import json
import logging
import os
import random
from collections import namedtuple
from os.path import join, exists
from typing import List, Dict
import config
import utils
from clf_distill_loss_functions import *
from train import *


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
