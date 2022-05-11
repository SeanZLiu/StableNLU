from typing import List, Dict, Iterable
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Dataset, \
    Sampler
from clf_distill_loss_functions import *
from utils import Processor, process_par
from train import TextPairExample

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, example_id, input_ids, segment_ids, label_id, bias):
        self.example_id = example_id
        self.input_ids = input_ids
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.bias = bias

# todo
class AutoExampleConverter(Processor):
    def __init__(self, max_seq_length, tokenizer):
        self.max_seq_length = max_seq_length
        self.tokenizer = tokenizer

    def process(self, data: Iterable):
        features = []
        tokenizer = self.tokenizer
        max_seq_length = self.max_seq_length
        for example in data:
            encoded_input = tokenizer(example.premise,example.hypothesis,truncation=True,max_length=max_seq_length,return_token_type_ids=True)
            features.append(
                InputFeatures(
                    example_id=example.id,
                    input_ids=np.array(encoded_input['input_ids']),
                    segment_ids=np.array(encoded_input['token_type_ids']),
                    label_id=example.label,
                    bias=None
                ))
        return features


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
