#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/2/28 上午10:09
# @Author  : Jingshuo Liu
# @File    : change_train

"""
# todo  按照新的AdamW进行了更新，但不知道会不会有问题，特别是step更新warmup
max_length 128 是否需要修改 batch32 是否需要修改

"""
from func.Opt import *
from func.Load import *
from func.Pre_process import *
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

import numpy as np
# from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.modeling import BertConfig, WEIGHTS_NAME, CONFIG_NAME
# from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear
from pytorch_pretrained_bert.tokenization import BertTokenizer

from transformers import AutoTokenizer,AutoConfig
from transformers import RobertaTokenizer
from transformers import RobertaConfig
from transformers import AdamW, get_linear_schedule_with_warmup


from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Dataset, \
    Sampler
from tqdm import trange, tqdm

import config
import utils

import clf_distill_loss_functions
from bert_distill import BertDistill, MyRoberta, MyErnie
from clf_distill_loss_functions import *

from predictions_analysis import visualize_predictions
from utils import Processor, process_par
import pickle
# Its a hack, but I didn't want a tensorflow dependency be required to run this code, so
# for now we just copy-paste MNLI loading stuff from debias/datasets/mnli
# todo
def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0 - x


HANS_URL = "https://raw.githubusercontent.com/tommccoy1/hans/master/heuristics_evaluation_set.txt"

NLI_LABELS = ["contradiction", "entailment", "neutral"]
NLI_LABEL_MAP = {k: i for i, k in enumerate(NLI_LABELS)}
REV_NLI_LABEL_MAP = {i: k for i, k in enumerate(NLI_LABELS)}
NLI_LABEL_MAP["hidden"] = NLI_LABEL_MAP["entailment"]

TextPairExample = namedtuple("TextPairExample", ["id", "premise", "hypothesis", "label"])



def main():

    parser = argparse.ArgumentParser(
        description='train.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    model_opts(parser)
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
        torch.manual_seed(args.seed)
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
    if args.model_type == 'bert':
        if "-uncased" in args.bert_model:
            do_lower_case = True
        elif "-cased" in args.bert_model:
            do_lower_case = False
        else:
            raise NotImplementedError(args.bert_model)

    # todo
    if args.model_type == 'bert':
        tokenizer = BertTokenizer.from_pretrained(args.bert_model,
                                              do_lower_case=do_lower_case)
    elif args.model_type == 'ernie':
        tokenizer = AutoTokenizer.from_pretrained( "nghuyong/ernie-2.0-en", cache_dir='./ernie_cache')
    else:
        tokenizer = RobertaTokenizer.from_pretrained("/users5/kxiong/huggingface_transformers/roberta-base")

    num_train_optimization_steps = None
    train_examples = None
    if args.do_train:
        train_examples = load_mnli(True, args.debug_num if args.debug else None)
        num_train_optimization_steps = int(
            len(
                train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
        if args.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()
        loss_fn.num_train_optimization_steps = int(num_train_optimization_steps)
        loss_fn.num_epochs = int(args.num_train_epochs)

    # Prepare model
    # todo
    # cache_dir = args.cache_dir if args.cache_dir else os.path.join(
    #     str(PYTORCH_PRETRAINED_BERT_CACHE),
    #     'distributed_{}'.format(args.local_rank))
    if args.model_type == 'bert':
        model = BertDistill.from_pretrained(
        args.bert_model, num_labels=3, loss_fn=loss_fn)
    elif args.model_type == 'ernie':
        model = MyErnie.from_pretrained("nghuyong/ernie-2.0-en", num_labels=3, loss_fn=loss_fn)
    else:
        model = MyRoberta.from_pretrained(pretrained_model_name_or_path="/users5/kxiong/huggingface_transformers/roberta-base", num_labels=3,loss_fn=loss_fn)
    # todo

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
        # todo 修改BertAdam为AdamW
        #optimizer = BertAdam(optimizer_grouped_parameters,
        #                      lr=args.learning_rate,
        #                      warmup=args.warmup_proportion,
        #                      t_total=num_train_optimization_steps)

        optimizer = AdamW(
            optimizer_grouped_parameters, lr=args.learning_rate, correct_bias=False
        )  # To reproduce BertAdam specific behavior set correct_bias=False
        if args.do_train:
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=float(args.warmup_proportion) * num_train_optimization_steps, num_training_steps=num_train_optimization_steps
            )  # PyTorch scheduler
        # todo 修改至此

    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0

    if args.do_train:
        # todo 预处理函数变更
        if args.model_type=='bert':
            train_features: List[InputFeatures] = convert_examples_to_features(
            train_examples, args.max_seq_length, tokenizer, args.n_processes)
        else:
            train_features: List[InputFeatures] = convert_examples_to_auto_features(
                train_examples, args.max_seq_length, tokenizer, args.n_processes)

        if args.which_bias == "mix":
            hypo_bias_map = load_bias("hypo")
            hans_bias_map = load_bias("hans")
            bias_map = {}
            def compute_entropy(probs, base=3):
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

                # todo 不需修改
                logits, loss = model(input_ids, segment_ids, input_mask, label_ids, bias,
                                     teacher_probs)
                # todo

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

                    # todo 除optimizer更新外，另加两行操作
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), 1.0
                    )  # Gradient clipping is not in AdamW anymore (so you can use amp without issue)
                    optimizer.step()
                    scheduler.step()

                    # todo

                    optimizer.zero_grad()
                    global_step += 1

        # Save a trained model and the associated configuration
        model_to_save = model.module if hasattr(model,
                                                'module') else model  # Only save the model it-self
        output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
        torch.save(model_to_save.state_dict(), output_model_file)
        output_config_file = os.path.join(output_dir, CONFIG_NAME)

        # todo
        if args.model_type != 'bert':
            output_config_file = os.path.join(output_dir, 'config.json')
        # todo

        with open(output_config_file, 'w') as f:
            f.write(model_to_save.config.to_json_string())

        # Record the args as well
        arg_dict = {}
        for arg in vars(args):
            arg_dict[arg] = getattr(args, arg)
        with open(join(output_dir, "args.json"), 'w') as out_fh:
            json.dump(arg_dict, out_fh)

        # Load a trained model and config that you have fine-tuned
        if args.model_type == 'bert':
            config = BertConfig(output_config_file)
            model = BertDistill(config, num_labels=3, loss_fn=loss_fn)
        elif args.model_type == 'ernie':
            config = AutoConfig.from_pretrained(output_config_file)
            model = MyErnie(config,num_labels=3, loss_fn=loss_fn)
        else:
            config = RobertaConfig(output_config_file)
            model = MyRoberta(config,num_labels=3, loss_fn=loss_fn)
        model.load_state_dict(torch.load(output_model_file))
    else:
        if args.model_type == 'bert':
            output_config_file = os.path.join(output_dir, CONFIG_NAME)
            config = BertConfig.from_pretrained(output_config_file)
            model = BertDistill(config, num_labels=3, loss_fn=loss_fn)
        elif args.model_type == 'ernie':
            config = AutoConfig.from_pretrained(os.path.join(output_dir, 'config.json'))
            model = MyErnie(config,num_labels=3, loss_fn=loss_fn)
        else:
            config = RobertaConfig.from_pretrained(os.path.join(output_dir, 'config.json'))
            model = MyRoberta(config,num_labels=3, loss_fn=loss_fn)
        output_model_file = os.path.join(output_dir, WEIGHTS_NAME)

        model.load_state_dict(torch.load(output_model_file))

    model.to(device)

    if not args.do_eval and not args.do_test:
        return
    if not (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        return

    model.eval()

    if args.do_eval:
        eval_datasets = [("mnli_dev_m", load_mnli(False)),
                         ("mnli_dev_mm", load_mnli(False, custom_path="dev_mismatched.tsv"))]
        eval_datasets += [("hans", load_hans())]
        eval_datasets += load_hans_subsets()
        if args.do_eval_on_train:
            eval_datasets += [("mnli_train", load_mnli(True))]
    else:
        eval_datasets = []

    if args.get_bert_output: # used to generate shallow feature file
        eval_datasets = [("mnli", load_mnli(is_train=True))]

    if args.do_test:
        test_datasets = load_all_test_jsonl()
        eval_datasets += test_datasets
        subm_paths = ["../submission/{}.csv".format(x[0]) for x in test_datasets]

    for ix, (name, eval_examples) in enumerate(eval_datasets):
        logging.info("***** Running evaluation on %s *****" % name)
        logging.info("  Num examples = %d", len(eval_examples))
        logging.info("  Batch size = %d", args.eval_batch_size)
        if args.model_type=='bert':
            eval_features = convert_examples_to_features(
                eval_examples, args.max_seq_length, tokenizer)
        else:
            eval_features = convert_examples_to_auto_features(
                eval_examples, args.max_seq_length, tokenizer)
        eval_features.sort(key=lambda x: len(x.input_ids))
        all_label_ids = np.array([x.label_id for x in eval_features])
        eval_dataloader = build_eval_dataloader(eval_features, args.eval_batch_size)

        eval_loss = 0
        nb_eval_steps = 0
        probs = []
        test_subm_ids = []
        if args.get_bert_output:
            # todo 新增list用于存储bert输出 现在要改成新建或读取已存在的词典，因为id要与其一一对应
            output_list = {}
            if exists(args.shallow_feature_file):
                with open(args.shallow_feature_file,'rb') as shallow_f:
                    output_list = pickle.load(shallow_f)  # 读入字典，向后面继续添加一行
        if args.get_logits:
            # todo 新增list用于存储logits输出
            logits_list = {}
            if exists(args.logits_file):
                with open(args.logits_file,'rb') as logits_f:
                    logits_list = pickle.load(logits_f)

        for example_ids, input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader,
                                                                               desc="Evaluating",
                                                                               ncols=100):
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            label_ids = label_ids.to(device)

            with torch.no_grad():
                # todo 不需修改
                logits = model(input_ids, segment_ids, input_mask)

                lgts = logits.cpu().numpy()
                if args.get_logits:  # 记录logits
                    exm_id1 = example_ids.cpu().numpy()
                    for i, sample_id in enumerate(exm_id1):
                        if sample_id not in logits_list:
                            logits_list[sample_id] = []
                        #if sample_id not in set_this_time:
                        logits_list[sample_id].append(lgts[i])

                # todo 可以输出bert后的特征向量表示，转为cpu形式后存储在list中，随后用pickle保存到文件里
                if args.get_bert_output:  # 如果需要bert的CLS输出
                    pooled_output = model.get_bert_output(input_ids, segment_ids, input_mask).data
                    pooled_output = pooled_output.cpu().numpy()
                    exm_id = example_ids.cpu().numpy()
                    for i, sample_id in enumerate(exm_id):
                        if sample_id not in output_list:
                            output_list[sample_id] = []

                        output_list[sample_id].append(pooled_output[i])

            # create eval loss and other metric required by the task
            loss_fct = CrossEntropyLoss()
            tmp_eval_loss = loss_fct(logits.view(-1, 3), label_ids.view(-1))

            eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            probs.append(torch.nn.functional.softmax(logits, 1).detach().cpu().numpy())
            test_subm_ids.append(example_ids.cpu().numpy())

        if args.get_bert_output:
            # todo 将list写入到文件中
            logging.info("********************************")
            logging.info(len(output_list[list(output_list.keys())[0]]))  # 取一个样本 提示输出现在有多少行了
            logging.info(len(logits_list[list(logits_list.keys())[0]]))  # 取一个样本 提示输出现在有多少行了
            logging.info("********************************")
            #print(len(set_this_time))
            pickle.dump(output_list,open(args.shallow_feature_file,"wb"))
            pickle.dump(logits_list,open(args.logits_file,"wb"))
        probs = np.concatenate(probs, 0)
        test_subm_ids = np.concatenate(test_subm_ids, 0)
        eval_loss = eval_loss / nb_eval_steps

        if "hans" in name:
            # take max of non-entailment rather than taking their sum
            probs[:, 0] = probs[:, [0, 2]].max(axis=1)
            # probs[:, 0] = probs[:, 0] + probs[:, 2]
            probs = probs[:, :2]

        preds = np.argmax(probs, axis=1)

        result = {"acc": simple_accuracy(preds, all_label_ids)}
        result["loss"] = eval_loss

        conf_plot_file = os.path.join(output_dir, "eval_%s_confidence.png" % name)
        ECE, bins_acc, bins_conf, bins_num = visualize_predictions(probs, all_label_ids, conf_plot_file=conf_plot_file)
        result["ECE"] = ECE
        result["bins_acc"] = bins_acc
        result["bins_conf"] = bins_conf
        result["bins_num"] = bins_num

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
