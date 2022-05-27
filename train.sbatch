#!/bin/bash
#SBATCH -J train                        
#SBATCH -p compute                            
#SBATCH -N 1                                 
#SBATCH --cpus-per-task=4                    
#SBATCH --mem-per-cpu=8G
#SBATCH -t 36:00:00                          
#SBATCH --gres=gpu:tesla_p100-pcie-16gb:1      

source ~/.bashrc

conda activate /users5/kxiong/miniconda3/envs/py36_jsliu

cd /users5/kxiong/jsliu/emnlp2020-debiasing-unknown/src 

#for file_name in 135 246 357 468 579 680 791 802 913
#do
#CUDA_VISIBLE_DEVICES=0 python get_shallow_output_full_tasks.py \
# --output_dir ../experiments_shallow_mnli/bert_base_sampled2K_epoch5_seed$file_name \
#  --do_eval --do_eval_on_train --mode none\
# --seed 111 --which_bias hans --task mnli \
#   --get_bert_output --shallow_feature_file shallow_features/shallow_mnli_train_5epoch_new.pkl \
#  --get_logits --logits_file logits_files/logits_mnli_train_5epoch_new.pkl
#done

#for shallow_num in 1 3 5 7 9 
#do
#CUDA_VISIBLE_DEVICES=0 python train_bert_features.py \
# --output_dir ../experiments_self_debias_mnli_seed111_different_shallow/bert_seed791_5epochshallow_$shallow_num \
#  --do_train  --mode none --custom_teacher ../teacher_preds/mnli_trained_on_sample2K_seed111.json\
# --seed 791 --which_bias hans --task mnli --shallow_model_num $shallow_num --shallow_feature_file shallow_features/shallow_mnli_train_5epoch_new.pkl
#done
#
#for shallow_num in 1 3 5 7 9
#do
#CUDA_VISIBLE_DEVICES=0  python train_distill_bert.py \
# --output_dir ../experiments_self_debias_mnli_seed111_different_shallow/bert_seed791_5epochshallow_$shallow_num \
#  --do_eval --mode none --custom_teacher ../teacher_preds/mnli_trained_on_sample2K_seed111.json\
# --seed 791 --which_bias hans
#done

#for shallow_num in 1 3 5 7 9
#do
#CUDA_VISIBLE_DEVICES=0 python train_roberta_distill.py \
#  --train_with_feature --model_type bert --output_dir ../experiments_self_debias_mnli_seed111_different_shallow/ernie/bert_seed135_ernieshallow_$shallow_num \
#  --do_train --do_eval --mode none --custom_teacher ../teacher_preds/mnli_trained_on_sample2K_seed111.json\
# --seed 135 --which_bias hans --task mnli --shallow_model_num $shallow_num --shallow_feature_file shallow_features/shallow_mnli_train_5epoch_ernie.pkl
#done

CUDA_VISIBLE_DEVICES=0 python train_roberta_distill.py \
  --train_with_feature --model_type bert --output_dir ../experiments_self_debias_mnli_seed111_different_shallow/bert_prune_shallow_seed135 \
  --do_train --do_eval --mode none --custom_teacher ../teacher_preds/mnli_trained_on_sample2K_seed111.json\
 --seed 135 --which_bias hans --task mnli --shallow_model_num 1 --shallow_feature_file shallow_features/prune_feature.pkl

CUDA_VISIBLE_DEVICES=0 python train_roberta_distill.py \
  --model_type bert --output_dir ../experiments_self_debias_mnli_seed111_different_shallow/bert_without_prune_shallow_seed135 \
  --do_train --do_eval --mode none --custom_teacher ../teacher_preds/mnli_trained_on_sample2K_seed111.json\
 --seed 135 --which_bias hans --task mnli 
