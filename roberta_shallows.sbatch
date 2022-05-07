#!/bin/bash
#SBATCH -J train                        
#SBATCH -p compute                            
#SBATCH -o roberta_shallows_new.out
#SBATCH -N 1                                 
#SBATCH --cpus-per-task=4                    
#SBATCH --mem-per-cpu=8G
#SBATCH -t 36:00:00                          
#SBATCH --gres=gpu:tesla_p100-pcie-16gb:1      

source ~/.bashrc

conda activate /users5/kxiong/miniconda3/envs/py36_jsliu

cd /users5/kxiong/jsliu/emnlp2020-debiasing-unknown/src
# 训练shallow model，注意对Roberta学习率需要设为2e-5
for this_seed in 111 222 333 444 777 999 112 334 556 667 778 889 267 391 801
do
CUDA_VISIBLE_DEVICES=0 python train_roberta_distill.py \
  --model_type roberta --output_dir ../experiments_shallow_mnli/roberta/roberta_base_sampled2K_seed$this_seed \
  --do_train --do_eval --do_eval_on_train --mode none\
  --seed $this_seed --which_bias hans --debug --num_train_epochs 5 --debug_num 2000 --learning_rate 1e-5
done
