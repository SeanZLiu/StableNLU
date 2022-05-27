#!/bin/bash
#SBATCH -J train                        
#SBATCH -p compute                            
#SBATCH -o ernie_t.out
#SBATCH -N 1                                 
#SBATCH --cpus-per-task=4                    
#SBATCH --mem-per-cpu=8G
#SBATCH -t 36:00:00                          
#SBATCH --gres=gpu:tesla_p100-pcie-16gb:1      

source ~/.bashrc

conda activate /users5/kxiong/miniconda3/envs/py36_jsliu

cd /users5/kxiong/jsliu/emnlp2020-debiasing-unknown/src

#for this_seed in 111 222 333 444 555 666 777 888 999
#do
#CUDA_VISIBLE_DEVICES=0 python train_roberta_distill.py \
#  --model_type ernie --output_dir ../experiments_shallow_mnli/ernie/ernie_base_sampled2K_5epoch_$this_seed \
#  --do_train --do_eval --do_eval_on_train --mode none\
#  --seed $this_seed --which_bias hans --debug --num_train_epochs 5 --debug_num 2000 --learning_rate 3e-5 
#done


# 生成shallow feature文件
#for this_seed in 111 222 333 444 555 666 777 888 999 
#do
#CUDA_VISIBLE_DEVICES=0 python train_roberta_distill.py \
# --model_type ernie --output_dir ../experiments_shallow_mnli/ernie/ernie_base_sampled2K_5epoch_$this_seed \
#  --do_eval --do_eval_on_train --mode none\
# --seed 111 --which_bias hans  \
#   --get_bert_output --shallow_feature_file shallow_features/shallow_mnli_train_5epoch_ernie.pkl \
#  --get_logits --logits_file logits_files/logits_mnli_train_5epoch_ernie.pkl
#done

# 训练main model并eval，注意这里用的是bert得到的shallow feature文件

#for shallow_num in 1 3 5 7 9
#do
#CUDA_VISIBLE_DEVICES=0 python train_roberta_distill.py \
#  --train_with_feature --model_type ernie --output_dir ../experiments_self_debias_mnli_seed111_different_shallow/ernie_main/ernie_seed555_bert_$shallow_num \
#  --do_train --do_eval --mode none --custom_teacher ../teacher_preds/mnli_trained_on_sample2K_seed111.json\
# --seed 555 --which_bias hans --task mnli --shallow_model_num $shallow_num --shallow_feature_file shallow_features/shallow_mnli_train_5epoch.pkl --learning_rate 5e-5
#done
for shallow_seed in 555 777
do
CUDA_VISIBLE_DEVICES=0 python train_roberta_distill.py \
  --model_type ernie --output_dir ../experiments_self_debias_mnli_seed111_different_shallow/ernie_main/ernie_baseline_$shallow_seed \
  --do_train --do_eval --mode none --custom_teacher ../teacher_preds/mnli_trained_on_sample2K_seed111.json\
 --seed $shallow_seed --which_bias hans --task mnli
done
