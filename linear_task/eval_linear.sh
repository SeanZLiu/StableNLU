#!/bin/bash
# 用ernie生成hans的feature格式文件 用于linear的评估
#for file_name in 111 222 333 444 555 666 777 888 999
#do
#CUDA_VISIBLE_DEVICES=0 python train_roberta_distill.py \
# --model_type ernie --output_dir ../experiments_shallow_mnli/ernie/ernie_base_sampled2K_5epoch_$file_name \
#  --do_eval --gene_challenge --mode none\
# --seed 111 --which_bias hans \
#   --get_bert_output --shallow_feature_file shallow_features/shallow_mnli_hans_ernie.pkl \
#  --get_logits --logits_file logits_files/logits_mnli_hans_ernie.pkl
#done

# 用ernie生成dev的feature格式文件
#for file_name in 111 222 333 444 555 666 777 888 999
#do
#CUDA_VISIBLE_DEVICES=0 python train_roberta_distill.py \
# --model_type ernie --output_dir ../experiments_shallow_mnli/ernie/ernie_base_sampled2K_5epoch_$file_name \
#  --do_eval --mode none\
# --seed 111 --which_bias hans \
#   --get_bert_output --shallow_feature_file shallow_features/shallow_mnli_dev_ernie.pkl \
#  --get_logits --logits_file logits_files/logits_mnli_dev_ernie.pkl
#done

# 用dev和hans来eval训练好的linear model
for shallow_num in 1 3 5 7 9
do
CUDA_VISIBLE_DEVICES=0 python train_bert_features.py \
  --linear --output_dir linear_net/linear_seed357_ernie_shallow$shallow_num \
  --do_eval  --mode none --custom_teacher ../teacher_preds/mnli_trained_on_sample2K_seed111.json\
 --seed 357 --which_bias hans --task mnli --shallow_model_num $shallow_num --shallow_feature_file shallow_features/shallow_mnli_train_5epoch.pkl
done
