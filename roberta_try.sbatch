#!/bin/bash
#SBATCH -J train                        
#SBATCH -p compute                            
#SBATCH -o roberta_try.out
#SBATCH -N 1                                 
#SBATCH --cpus-per-task=4                    
#SBATCH --mem-per-cpu=8G
#SBATCH -t 36:00:00                          
#SBATCH --gres=gpu:tesla_p100-pcie-16gb:1      

source ~/.bashrc

conda activate /users5/kxiong/miniconda3/envs/py36_jsliu

cd /users5/kxiong/jsliu/emnlp2020-debiasing-unknown/src
# 训练shallow model，注意对Roberta学习率需要设为2e-5
CUDA_VISIBLE_DEVICES=0 python train_roberta_distill.py \
  --model_type roberta --output_dir ../experiments_shallow_mnli/roberta/roberta_base_sampled2K_seed222_6epoch \
  --do_train --do_eval --do_eval_on_train --mode none\
  --seed 222 --which_bias hans --debug --num_train_epochs 6 --debug_num 2000 --learning_rate 2e-5

# 生成shallow feature文件
CUDA_VISIBLE_DEVICES=0 python train_roberta_distill.py \
 --output_dir ../experiments_shallow_mnli/roberta/roberta_base_sampled2K_seed222_6epoch \
  --do_eval --do_eval_on_train --mode none\
 --seed 111 --which_bias hans  \
   --get_bert_output --shallow_feature_file shallow_features/shallow_mnli_train_6epoch_roberta.pkl \
  --get_logits --logits_file logits_files/logits_mnli_train_6epoch_roberta.pkl

# 训练main model
CUDA_VISIBLE_DEVICES=0 python train_bert_features.py \
 --output_dir ../experiments_self_debias_mnli_seed111_different_shallow/roberta/bert_seed791_6epochshallow_1 \
  --do_train  --mode none --custom_teacher ../teacher_preds/mnli_trained_on_sample2K_seed111.json\
 --seed 791 --which_bias hans --task mnli --shallow_model_num 1 --shallow_feature_file shallow_features/shallow_mnli_train_6epoch_roberta.pkl

#用hans对模型性能进行评估
CUDA_VISIBLE_DEVICES=0  python train_distill_bert.py \
 --output_dir ../experiments_self_debias_mnli_seed111_different_shallow/roberta/bert_seed791_6epochshallow_1 \
  --do_eval --mode none --custom_teacher ../teacher_preds/mnli_trained_on_sample2K_seed111.json\
 --seed 791 --which_bias hans
