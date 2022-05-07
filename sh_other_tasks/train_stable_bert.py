"""
Script to train BERT on MNLI with our loss function

Modified from the old "run_classifier" script from
https://github.com/huggingface/pytorch-transformer
"""

import argparse
import json
import logging
import os
import random
from collections import namedtuple
from os.path import join, exists
from typing import List, Dict, Iterable
import time
# temporary hack for the pythonroot issue
import sys

import numpy as np
from torch.nn import functional as F
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.modeling import BertConfig, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.modeling import BertForSequenceClassification
from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.tokenization import BertTokenizer
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Dataset, \
    Sampler
from torch.autograd import Variable
from torch.distributions.multivariate_normal import MultivariateNormal
from tqdm import trange, tqdm
#import collections.abc as container_abc

from onmt import config
from onmt.Utils import *
from onmt.Opts import *
from onmt.Visualization_analysis import *

from onmt.bert_distill import BertDistill
from onmt.clf_distill_loss_functions import *

sys.path.append("/users4/ldu/git_clones/apex_u/apex-master/")
from apex import amp

import pdb
import copy

NLI_LABELS = ["contradiction", "entailment", "neutral"]
NLI_LABEL_MAP = {k: i for i, k in enumerate(NLI_LABELS)}
REV_NLI_LABEL_MAP = {i: k for i, k in enumerate(NLI_LABELS)}
NLI_LABEL_MAP["hidden"] = NLI_LABEL_MAP["entailment"]

TextPairExample = namedtuple("TextPairExample", ["id", "premise", "hypothesis", "label"])

parser = argparse.ArgumentParser(
    description='Train.py',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

model_opts(parser)
args = parser.parse_args()

gpu_ls = [int(i) for i in args.gpu_ls]
add_stdout_logger()

output_dir = args.output_dir

#loss_fn = fetch_loss_fuction(args)
loss_fn = torch.nn.CrossEntropyLoss()   

if exists(output_dir):
    if len(os.listdir(output_dir)) > 0:
        logging.warning("Output dir exists and is non-empty")
else:
    os.makedirs(output_dir)

print("Saving model to %s" % output_dir)


if args.seed is not None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if len(args.gpu_ls) > 0:
        torch.cuda.manual_seed_all(args.seed)

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
    
if args.pret:
    train_examples = load_mnli(True, args.debug_num if args.debug else None)
    train_features: List[InputFeatures] = convert_examples_to_features(
        train_examples, args.max_seq_length, tokenizer, args.n_processes)
else:
    if 'mnli' in args.dataset:
        args.pret_features = "/users4/ldu/stable_nli/emnlp2020-debiasing-unknown/experiments_self_debias_mnli/matched_features/tot_n.pkl" 
    elif 'qqp' in args.dataset:
        args.pret_features = "/users4/ldu/stable_nli/emnlp2020-debiasing-unknown/dataset/final_features/qqp_train_features.pkl"
    elif 'fever' in args.dataset:
        args.pret_features = "/users4/ldu/stable_nli/emnlp2020-debiasing-unknown/dataset/final_features/fever_train_features.pkl"
    
    all_features = pickle.load(open(args.pret_features, 'rb'))
    if 'mnli' in args.pret_features:
        train_features = all_features['mnli_train']
    else:
        train_features = all_features

    if args.debug:
        train_features = random.sample(train_features, args.debug_num)

num_train_optimization_steps = int(
    (len(train_features) - args.sample_size_e) / args.train_batch_size) * args.num_train_epochs

loss_fn.num_train_optimization_steps = int(num_train_optimization_steps)
loss_fn.num_epochs = int(args.num_train_epochs)

# Prepare model
cache_dir = args.cache_dir if args.cache_dir else os.path.join(
    str(PYTORCH_PRETRAINED_BERT_CACHE))


model_stable = ini_from_pretrained(args.stable_model_path, de_novo=args.de_novo)
#model_shallow = ini_from_pretrained(args.shallow_model_path)
config_stable = model_stable.config

gpu_stable = gpu_ls[0]
gpu_shallow = gpu_ls[-1]
model_stable.to(gpu_stable)

#model_shallow.cuda(gpu_shallow)

# Prepare optimizer
parameters_stable = list(model_stable.named_parameters())
#parameters_shallow = list(model_shallow.named_parameters())
    
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

optimizer_grouped_parameters_stable = [
    {'params': [p for n, p in parameters_stable if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in parameters_stable if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
    
#optimizer_grouped_parameters_shallow = [
#    {'params': [p for n, p in parameters_shallow if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
#    {'params': [p for n, p in parameters_shallow if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
#]

optimizer_stable = BertAdam(optimizer_grouped_parameters_stable,lr=args.learning_rate,warmup=args.warmup_proportion,
                     t_total=num_train_optimization_steps)

#optimizer_shallow = BertAdam(optimizer_grouped_parameters_shallow,lr=args.learning_rate,warmup=args.warmup_proportion,
#                     t_total=num_train_optimization_steps)
                     
model_stable, optimizer_stable = amp.initialize(model_stable, optimizer_stable, opt_level="O1")
#model_shallow, optimizer_shallow = amp.initialize(model_shallow, optimizer_shallow, opt_level="O1")

#model_stable = nn.DataParallel(model_stable,  device_ids=gpu_ls)
#model_shallow = nn.DataParallel(model_shallow,  device_ids=gpu_ls[::-1])


global_step = 0
nb_tr_steps = 0
tr_loss = 0

logging.info("***** Running training *****")
logging.info("  Num examples = %d", len(train_features))
logging.info("  Batch size = %d", args.train_batch_size)
logging.info("  Num steps = %d", num_train_optimization_steps)

train_dataloader = build_train_dataloader(train_features[:-args.sample_size_e], args.train_batch_size,
                                          args.seed, args.sorted)
train_dataloader_random = build_eval_dataloader(train_features[:-args.sample_size_e], args.train_batch_size)
exam_dataloader = build_eval_dataloader(train_features[-args.sample_size_e:], args.train_batch_size)

loss_ema = 0
total_steps = 0
decay = 0.99
sample_size_c = 0
sample_size_e = 0

name = parse_opt_to_name(args)
print(name)
time_start = str(int(time.time()))[-6:]  

model_output_path = os.path.join(output_dir, name + time_start)
os.makedirs(model_output_path, exist_ok=True)

f = open('../records/' + name + '_' + time_start + '.csv', 'a+')
f.write(args.stable_model_path + '\n')
f.write(args.shallow_model_path + '\n')
f.close()    

for epoch in trange(int(args.num_train_epochs), desc="Epoch", ncols=100):
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    #pbar = tqdm(train_dataloader, desc="loss", ncols=100)
        
    model_stable.eval()
    #model_shallow.eval()
    if 'cfs' in args.mode:
        # if epoch > 0:
        # if epoch == 2:
        if epoch > 0:
            res_A = do_evaluation(model_stable, train_dataloader,  gpu_id=gpu_stable) # cls_score, pooled_out, eval_accuracy, ans
            #res_B = do_evaluation(model_stable, exam_dataloader, gpu_id=gpu_stable)
            res_B = res_A
            confused_pairs, train_dataloader = confused_pair_gen(res_A, res_B, args, train_features[:-args.sample_size_e]) 
            print('num_confused_pairs:', confused_pairs.shape[0])
    
    pbar = tqdm(train_dataloader, desc="loss", ncols=100)
    
    '''
    if False:
        if args.strategy_adv in ['paired_train', 'random_train']:
            res_B = do_evaluation(model_stable, train_dataloader_random,gpu_id=gpu_stable) # cls_score, pooled_out, eval_accuracy, ans
        elif args.strategy_adv == 'paired_exam':
            res_B = do_evaluation(model_stable, exam_dataloader, gpu_id=gpu_stable) # cls_score, pooled_out, eval_accuracy, ans
        
        f = open('../records/' + name + '_' + time_start + '.csv', 'a+')
        f.write(','.join(['res_B', str(res_B['acc'])]) + '\n') 
        f.close()  
        
        confused_pairs, confused_dataloader = confused_pair_gen(res_A['pooled_outs'], res_B['pooled_outs'], res_A['probs'], res_B['probs'], res_A['labels'], res_B['labels'], args.thre_posi, args.thre_nega, args.tau, args.uplim_candi_list, train_features[:-args.sample_size_e]) 

        confused_dataloader = copy.deepcopy(train_dataloader) # !!
        confused_dataloader = [l for l in confused_dataloader]
        
        num_confused_pair = len(confused_pairs)
        print(num_confused_pair)
            
        if num_confused_pair == 0:
            print('len(pairs_0) == 0 !!!')
        else:    
            print("Phase I: Update the Stable Model Using the Current Shallow Model")
    '''
    model_stable.train()
    train_dataloader_random = [l for l in train_dataloader_random]
    
    for step_I, batch in enumerate(pbar):
        batch = tuple(t.to(gpu_stable) for t in batch)
        
        if len(batch) == 5:
            example_ids, input_ids, input_mask, segment_ids, label_ids = batch
        elif len(batch) == 6:
            example_ids, input_ids, input_mask, segment_ids, label_ids, bias_features = batch
        #elif len(batch) == 7:
        #    example_ids, input_ids, input_mask, segment_ids, label_ids, bias_features, sample_weights = batch
        elif len(batch) == 7:
            example_ids, input_ids, input_mask, segment_ids, label_ids, bias_features, bias_probs = batch
        elif len(batch) > 7:
            print('wrong!')
        
        # pdb.set_trace()
        
        #logits_stable, pooled_out_stable = model_stable(input_ids, segment_ids, input_mask)
        logits, pooled_out_stable = model_stable(input_ids, segment_ids, input_mask)
        '''
        if args.strategy_adv in ['paired_train', 'paired_exam']:
            batch_adv = confused_dataloader[(step_I * (epoch + 1)) % len(confused_dataloader)]
            example_ids_adv, input_ids_adv, input_masks_adv, sentence_inds_adv, answers_adv, _ = batch_adv 
        elif args.strategy_adv == 'random_train':
            batch_adv = train_dataloader_random[(step_I * (epoch + 1)) % len(train_dataloader_random)]
            example_ids_adv, input_ids_adv, input_masks_adv, sentence_inds_adv, answers_adv,_ = batch_adv
        
        batch_adv = tuple(t.to(gpu_shallow) for t in batch_adv)    
        #logits_shallow, pooled_out_shallow = model_shallow(input_ids_adv, sentence_inds_adv, input_masks_adv)
        '''
        Lambda = args.Lambda

        if args.mode in ['mean', 'mean_regu', 'mean_bayes', 'mean_cfs']:
            bias_feature = bias_features.mean(dim=1) 
            # bias_feature = bias_features.mean(dim=1) * args.shallow_feature_weight
            # bias_feature = bias_features[:, 0, :] * args.shallow_feature_weight
            # bias_feature = bias_features[:,0,:]             
            # bias_feature = Variable(bias_feature, requires_grad=True)
            # bias_feature = bias_features[:,0,:]             
        elif args.mode == 'mean_aux':
            bias_feature = torch.tanh(model_stable.module.aux_linear(bias_features.mean(dim=1))) 

        elif args.mode in ['pod', 'pod_regu']:
            loss_fn = Pod(bias_probs)
            pooled_out = pooled_out_stable
        elif args.mode == 'pod_mean':
            loss_fn = Pod(bias_probs, avg=True)
            pooled_out = pooled_out_stable
        elif args.mode == 'exp_pod':
            loss_fn = ExpPod(bias_probs)
            pooled_out = pooled_out_stable
        elif args.mode == 'exp_pod_mean':
            loss_fn = ExpPod(bias_probs, avg=True)
            pooled_out = pooled_out_stable

        if args.mode == 'mean_bayes':
            bias_feature_var = bias_features.var(1) / bias_features.shape[0] + 0.1
            # Considering the instability of inverse operation for cov matrix, especially under the 
            # small sample size, here only the elements on the diagnoal are considered.
            classifier_parameter = list(model_stable.classifier.parameters())[0]
            
            stable_grad = gradient_calc(torch.softmax(logits, 1), label_ids, classifier_parameter, pooled_out_stable, mode=args.mode).sum(1) + 0.1
            stable_var = 1 / (stable_grad ** 2)

            pooled_out = pooled_out_stable  + (bias_feature / bias_feature_var) / (1 / stable_var + 1 / bias_feature_var)

            # pooled_out = (pooled_out_stable / stable_var + bias_feature / bias_feature_var) / (1 / stable_var + 1 / bias_feature_var)

        elif args.mode in ['mean', 'mean_cfs']:   
            # pooled_out = pooled_out_stable + bias_feature ## the original
            pooled_out = pooled_out_stable * bias_feature
            if args.mode == 'mean_cfs' and epoch == 0:
                pooled_out = pooled_out_stable + bias_feature ## the original
            else:
                pooled_out = pooled_out_stable + bias_feature

        elif args.mode in ['mean_rand', 'mean_cfs_rand']:   
            bias_feature_mean = bias_features.mean(dim=1)
            # bias_feature_var = bias_features.var(dim=1) + 0.001
            
            mask_tmp = torch.zeros_like(bias_feature_mean)
            # mask_tmp[:, :384] = 0.15
            mask_tmp[:, 384:] = 0.15
            mask_tmp = mask_tmp.to(mask_tmp.device)

            bias_feature_mean = bias_feature_mean + mask_tmp
            # bias_feature_mean = bias_feature_mean - mask_tmp

            pooled_out_tmp = pooled_out_stable + bias_feature_mean
            logits_sum_tmp = model_stable.classifier(pooled_out_tmp)
            classifier_parameter = list(model_stable.classifier.parameters())[0]
            thre = torch.FloatTensor([1])
            thre = thre.to(bias_probs.device)
            bias_feature_var = 1 / torch.max((gradient_calc(torch.softmax(logits_sum_tmp, 1), label_ids, classifier_parameter, pooled_out_tmp, mode=args.mode).sum(1)) ** 2, thre)
            
            m = MultivariateNormal(bias_feature_mean.cpu(), torch.diag_embed(bias_feature_var).cpu())
            bias_feature = m.rsample()
            bias_feature = bias_feature.to(bias_features.device)

            if args.mode == 'mean_cfs_rand' and epoch == 0:
                # pooled_out = pooled_out_stable ## the original
                pooled_out = pooled_out_stable + bias_feature
            else:
                pooled_out = pooled_out_stable + bias_feature

        elif args.mode == 'mean_addi':
            bias_feature = bias_features.mean(dim=1) 
            perm_id = torch.randperm(bias_features.shape[0])
            bias_feature_other = bias_features[perm_id].mean(dim=1)
            pooled_out = pooled_out_stable + bias_feature + 0.05 * bias_feature_other
            # pooled_out = pooled_out_stable + bias_feature + 0.2 * bias_feature_other
            # pooled_out = pooled_out_stable + bias_feature + 0.1 * bias_feature_other
            # pooled_out = pooled_out_stable + 0.9 * bias_feature + 0.1 * bias_feature_other
            # pdb.set_trace()
            # weights = torch.zeros_like(bias_probs.mean(1))
            # weights.to(bias_probs.device)
            # weights = weights.scatter(1, label_ids.unsqueeze(1), bias_probs.mean(1))
            # weights = weights.sum(1).unsqueeze(1)
            # pooled_out = pooled_out_stable + bias_feature * (1 - weights)
            # pooled_out = torch.tanh(pooled_out_stable + bias_feature)
            # pooled_out = torch.tanh(pooled_out_stable) + torch.tanh(bias_feature)

            # weights = torch.zeros_like(bias_probs)
            # weights.to(bias_probs.device)
            # weights = weights.scatter(2, label_ids.unsqueeze(1).expand(bias_features.shape[:2]).unsqueeze(2), bias_probs)
            # weights = weights.sum(2)
            # bias_feature = torch.matmul(bias_features.transpose(1, 2), weights.unsqueeze(2))
            # weights_sum = weights.sum(1)
            # bias_feature = bias_feature.squeeze() / weights_sum.unsqueeze(1)

            # pooled_out = pooled_out_stable
        elif args.mode == 'mean_regu':
            # pooled_out = torch.tanh(pooled_out_stable) + torch.tanh(bias_feature)
            pooled_out = torch.tanh(pooled_out_stable + bias_feature)

        elif args.mode == 'pod_regu':
            # loss_regu = -(bias_probs[:, 0, :] * torch.softmax(logits, 1)).sum()
            # loss_regu = -(bias_probs[:, 0, :] * torch.softmax(logits, 1)).log().sum()
            thre = torch.FloatTensor([0.9])
            thre = thre.to(bias_probs.device)
            
            loss_regu = - torch.min((bias_probs[:, 0, :] * torch.softmax(logits, 1)).sum(1), thre).sum()

        elif args.mode == 'mean_both':
            bias_feature = bias_features.mean(dim=1) * args.shallow_feature_weight
            pooled_out = pooled_out_stable + bias_feature ## the original
            pooled_out_sub = pooled_out_stable - bias_feature ## the original

        logits_sum = model_stable.classifier(pooled_out)

        loss_cls = loss_fn(logits_sum, label_ids)
        #if epoch > 0 and loss_cls < 0.01:
        #    pdb.set_trace()

        if args.mode == 'mean_regu':
            # bias_feature.retain_grad()
            # pooled_out_stable.retain_grad()
            # pooled_out.retain_grad()
            # if step_I < 1:
            logits_shallow = model_stable.classifier(bias_feature)

            classifier_parameter = list(model_stable.classifier.parameters())[0]
            # bias_grad = gradient_calc(torch.softmax(logits_sum, 1), label_ids, classifier_parameter, bias_feature)
            bias_grad = gradient_calc(torch.softmax(logits_shallow, 1), label_ids, classifier_parameter, bias_feature)
            
            # stable_grad = gradient_calc(torch.softmax(logits_sum, 1), label_ids, classifier_parameter, pooled_out_stable)
            stable_grad = gradient_calc(torch.softmax(logits, 1), label_ids, classifier_parameter, pooled_out_stable)
            tot_grad = gradient_calc(torch.softmax(logits_sum, 1), label_ids, classifier_parameter, pooled_out)
            # else:
            #     # bias_grad = bias_feature.grad
            #     # stable_grad = pooled_out_stable.grad
            # loss_regu = (bias_grad * stable_grad).sum() # 
            # loss_regu = ((bias_grad * stable_grad) ** 2).sum() # lst 3 (or torch.abs with lst 4)
            loss_regu = ((tot_grad ** 2) / (stable_grad ** 2) / (bias_grad) ** 2 ).sum() #lst. last 2: w/o the bias_grad term 
            
            loss = loss_cls + args.Lambda * loss_regu

        elif args.mode == 'pod_regu':
            loss = loss_cls + args.Lambda * loss_regu
        elif args.mode == 'mean_both':
            logits_sub = model_stable.classifier(pooled_out_sub)
            loss_cls_sub = loss_fn(logits_sub, label_ids)

            loss = loss_cls + 1 * loss_cls_sub

        else:
            loss = loss_cls
        
        total_steps += 1
        loss_ema = loss_ema * decay + loss.cpu().detach().numpy() * (1 - decay)
        descript = "loss=%.4f" % (loss_ema / (1 - decay ** total_steps))
        pbar.set_description(descript, refresh=False)
                    
        f = open('../records/' + name + '_' + time_start + '.csv', 'a+')
        if 'regu' in args.mode:
            f.write('phase1' + ',' + str(loss_cls.detach().cpu().numpy()) + ',' + str(loss_regu.detach().cpu().numpy()) + '\n')
        else:
            f.write('phase1' + ',' + str(loss_cls.detach().cpu().numpy()) + '\n')
        f.close() 
        
        #print(loss)  
        # optimizer_stable.step()
        # optimizer_stable.zero_grad()  

        with amp.scale_loss(loss, optimizer_stable) as scaled_loss:
            scaled_loss.backward()
        #loss.backward()
        optimizer_stable.step()
        optimizer_stable.zero_grad()  
        
        tr_loss += loss.item()
        nb_tr_examples += input_ids.size(0)
        nb_tr_steps += 1
    
        # optimizer_stable.zero_grad()
        global_step += 1
        
        sample_size_c += example_ids.shape[0]

    # Save a trained model and the associated configuration
    output_path_stable = os.path.join(model_output_path, 'stable_' + 'e_' + str(epoch) + WEIGHTS_NAME)
    if hasattr(model_stable, 'module'):
        torch.save(model_stable.module.state_dict(), output_path_stable)
    else:
        torch.save(model_stable.state_dict(), output_path_stable)

    #output_path_shallow = os.path.join(model_output_path, 'shallow_' + 'e_' + str(epoch) + WEIGHTS_NAME)
    #torch.save(model_shallow.state_dict(), output_path_shallow)
        
# Record the configs and args as well
output_config_file = os.path.join(model_output_path, CONFIG_NAME)
with open(output_config_file, 'w') as f:
    f.write(config_stable.to_json_string())

arg_dict = {}
for arg in vars(args):
    arg_dict[arg] = getattr(args, arg)
with open(join(model_output_path, "args.json"), 'w') as out_fh:
    json.dump(arg_dict, out_fh)


if args.do_eval:
    if not args.do_train:
        # Load a trained model and config that you have fine-tuned
        output_config_file = os.path.join(output_dir, CONFIG_NAME)
        config = BertConfig.from_json_file(output_config_file)
        output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
        model = BertDistill(config, num_labels=3)
        model.load_state_dict(torch.load(output_model_file))
    elif args.do_eval:
        model = model_stable.eval()
        del model_stable
    
        model.to(gpu_ls[0])
        
        model.eval()

    if 'mnli' in args.pret_features:
        if args.pret:
            eval_datasets = [("mnli_dev_m", load_mnli(False)),
                            ("mnli_dev_mm", load_mnli(False, custom_path="dev_mismatched.tsv"))]
        else:
            eval_datasets = []
            eval_datasets += [("mnli_dev_m", all_features["mnli_dev_m"])]
            eval_datasets += [("mnli_dev_mm", all_features["mnli_dev_m"])]
            
        eval_datasets += [("hans", load_hans())]
        eval_datasets += load_hans_subsets()
    elif 'qqp' in args.pret_features:
            eval_datasets = [('qqp_dev', pickle.load(open("/users4/ldu/stable_nli/emnlp2020-debiasing-unknown/dataset/final_features/qqp_dev_features.pkl", 'rb')))]
            eval_datasets += [('paws', pickle.load(open("/users4/ldu/stable_nli/emnlp2020-debiasing-unknown/dataset/final_features/paws_features.pkl", 'rb')))]
    elif 'fever' in args.pret_features:
            eval_datasets = [('fever_dev', pickle.load(open("/users4/ldu/stable_nli/emnlp2020-debiasing-unknown/dataset/final_features/fever_dev_features.pkl", 'rb')))]
            eval_datasets += [('symm', pickle.load(open("/users4/ldu/stable_nli/emnlp2020-debiasing-unknown/dataset/final_features/symm_features.pkl", 'rb')))]
    
    for ix, (name, eval_examples) in enumerate(eval_datasets):
        logging.info("***** Running evaluation on %s *****" % name)
        logging.info("  Num examples = %d", len(eval_examples))
        logging.info("  Batch size = %d", args.eval_batch_size)
        if 'mnli' in args.pret_features:
            if args.pret or 'hans' in name:
                eval_features = convert_examples_to_features(
                    eval_examples, args.max_seq_length, tokenizer)
            else:
                eval_features = all_features[name]
        else:
            eval_features = eval_examples

        eval_features.sort(key=lambda x: len(x.input_ids))
        all_label_ids = np.array([x.label_id for x in eval_features])
        eval_dataloader = build_eval_dataloader(eval_features, 128)
        
        for use_bias_feature in [False, True]:        
            if use_bias_feature and 'hans' in name:
                pass
            else:
                result = do_evaluation(model, eval_dataloader, gpu_ls[0], name, use_bias_feature=use_bias_feature, mode=args.mode, weight=args.shallow_feature_weight)
            
                name = name + '_' + str(use_bias_feature)[0]    
                output_eval_file = os.path.join(model_output_path, "eval_%s_results.txt" % name)
                output_all_eval_file = os.path.join(model_output_path, "eval_all_results.txt")
                with open(output_eval_file, "w") as writer, open(output_all_eval_file, "a") as all_writer:
                    logging.info("***** Eval results *****")
                    all_writer.write("eval results on %s:\n" % name)
                    
                    logging.info("  %s = %s", 'acc', str(result['acc']))
                    writer.write("%s = %s\n" % ('acc', str(result['acc'])))
                    all_writer.write("%s = %s\n" % ('acc', str(result['acc'])))
                
                output_file = os.path.join(model_output_path, "eval_res_%s.pkl" % name)
                pickle.dump(result, open(output_file, 'wb'))


