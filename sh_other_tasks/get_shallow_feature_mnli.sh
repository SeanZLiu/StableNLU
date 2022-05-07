#!/bin/bash

for file_name in 111 222 333 444 555 666 777 888 999 132 243 354 465 576 687
do
CUDA_VISIBLE_DEVICES=0 python get_shallow_output_full_tasks.py \
 --output_dir ../experiments_shallow_mnli/bert_base_sampled2K_seed$file_name \
  --do_eval --gene_challenge --mode none\
 --seed 111 --which_bias hans --task mnli \
 --get_bert_output --shallow_feature_file shallow_features/shallow_hans.pkl
done

CUDA_VISIBLE_DEVICES=0 python get_final_features.py \
 --output_dir ../experiments_self_debias_mnli_seed111/bert_reweighted_sampled2K_teacher_seed111_annealed_1to08 \
  --do_eval --gene_challenge --mode none\
 --seed 111 --which_bias hans --task mnli \
 --get_bert_output --shallow_feature_file shallow_features/shallow_hans.pkl \
 --final_feature final_features/hans_features.pkl

#for file_name in 111 222 333 444 555 666 777 888 999 132 243 354 465 576 687
#do
#CUDA_VISIBLE_DEVICES=0 python get_shallow_output_full_tasks.py \
# --output_dir ../experiments_shallow_mnli/bert_base_sampled2K_seed$file_name \
#  --do_eval  --mode none\
# --seed 111 --which_bias hans --task mnli \
# --get_bert_output --shallow_feature_file shallow_features/shallow_mnli_dev.pkl
#done

#CUDA_VISIBLE_DEVICES=0 python get_final_features.py \
# --output_dir ../experiments_self_debias_mnli_seed111/bert_reweighted_sampled2K_teacher_seed111_annealed_1to08 \
#  --do_eval  --mode none\
# --seed 111 --which_bias hans --task mnli \
# --get_bert_output --shallow_feature_file shallow_features/shallow_mnli_dev.pkl \
# --final_feature final_features/mnli_dev_features.pkl

for file_name in 111 222 333 444 555 666 777 888 999 132 243 354 465 576 687
do
CUDA_VISIBLE_DEVICES=0 python get_shallow_output_full_tasks.py \
 --output_dir ../experiments_shallow_mnli/bert_base_sampled2K_seed$file_name \
  --do_eval --do_eval_on_train --mode none\
 --seed 111 --which_bias hans --task mnli \
   --get_bert_output --shallow_feature_file shallow_features/shallow_mnli_train.pkl
done

CUDA_VISIBLE_DEVICES=0 python get_final_features.py \
 --output_dir ../experiments_self_debias_mnli_seed111/bert_reweighted_sampled2K_teacher_seed111_annealed_1to08 \
  --do_eval --do_eval_on_train --mode none\
 --seed 111 --which_bias hans --task mnli \
   --get_bert_output --shallow_feature_file shallow_features/shallow_mnli_train.pkl \
    --final_feature final_features/mnli_train_features.pkl
