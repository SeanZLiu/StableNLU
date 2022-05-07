#!/bin/bash


for file_name in 135 246 357 468 579 680 791 802 913
do
CUDA_VISIBLE_DEVICES=0 python get_shallow_output_full_tasks.py \
 --output_dir ../experiments_shallow_mnli/bert_base_sampled2K_epoch5_seed$file_name \
  --do_eval --do_eval_on_train --mode none\
 --seed 111 --which_bias hans --task mnli \
   --get_bert_output --shallow_feature_file shallow_features/shallow_mnli_train_5epoch_new.pkl \
  --get_logits --logits_file logits_files/logits_mnli_train_5epoch_new.pkl
done

#for file_name in 111 222 333 444 555  
#do
#CUDA_VISIBLE_DEVICES=0 python get_shallow_output_full_tasks.py \
# --output_dir ../experiments_shallow_mnli/bert_base_sampled2K_epoch5_seed$file_name \
#  --do_eval --do_eval_on_train --mode none\
# --seed 111 --which_bias hans --task mnli \
#   --get_bert_output --shallow_feature_file shallow_features/shallow_mnli_train_5epoch.pkl
#done

#CUDA_VISIBLE_DEVICES=0 python get_final_features.py \
# --output_dir ../experiments_self_debias_mnli_seed111/bert_reweighted_sampled2K_teacher_seed111_annealed_1to08 \
#  --do_eval --do_eval_on_train --mode none\
# --seed 111 --which_bias hans --task mnli \
#   --get_bert_output --shallow_feature_file shallow_features/shallow_mnli_train_5epoch.pkl \
#    --final_feature final_features/mnli_train_features_5epoch.pkl
