
for file_name in 111 222 333 444 555 666 679 777 888 999
do
CUDA_VISIBLE_DEVICES=0 python get_shallow_output_full_tasks.py \
 --output_dir ../experiments_shallow_qqp/bert_base_sampled500_seed$file_name \
  --do_eval  --mode none\
 --seed 111 --which_bias qqp_hans_json --task qqp \
 --get_bert_output --shallow_feature_file shallow_features/shallow_qqp_dev.pkl --custom_teacher ../teacher_preds/qqp_teacher_seed222.json
done

CUDA_VISIBLE_DEVICES=0 python get_final_features.py \
 --output_dir ../experiments_self_debias_qqp_seed777/bert_reweighted_sampled2K_teacher_seed666_annealed \
  --do_eval  --mode none\
 --seed 111 --which_bias qqp_hans_json --task qqp \
 --get_bert_output --shallow_feature_file shallow_features/shallow_qqp_dev.pkl \
 --final_feature final_features/qqp_dev_features.pkl --custom_teacher ../teacher_preds/qqp_teacher_seed222.json

for file_name in 111 222 333 444 555 666 679 777 888 999
do
CUDA_VISIBLE_DEVICES=0 python get_shallow_output_full_tasks.py \
 --output_dir ../experiments_shallow_qqp/bert_base_sampled500_seed$file_name \
  --do_eval --do_eval_on_train --mode none\
 --seed 111 --which_bias qqp_hans_json --task qqp \
   --get_bert_output --shallow_feature_file shallow_features/shallow_qqp_train.pkl --custom_teacher ../teacher_preds/qqp_teacher_seed222.json
done


for file_name in 111 222 333 444 555 666 679 777 888 999
do
CUDA_VISIBLE_DEVICES=0 python get_shallow_output_full_tasks.py \
 --output_dir ../experiments_shallow_qqp/bert_base_sampled500_seed$file_name \
  --do_eval --gene_challenge --mode none\
 --seed 111 --which_bias qqp_hans_json --task qqp \
 --get_bert_output --shallow_feature_file shallow_features/shallow_paws.pkl --custom_teacher ../teacher_preds/qqp_teacher_seed222.json
done

UDA_VISIBLE_DEVICES=0 python get_final_features_qqp_train.py \
 --output_dir ../experiments_self_debias_qqp_seed777/bert_reweighted_sampled2K_teacher_seed666_annealed \
  --do_eval --do_eval_on_train --mode none\
 --seed 111 --which_bias qqp_hans_json --task qqp \
   --get_bert_output --shallow_feature_file shallow_features/shallow_qqp_train.pkl \
    --final_feature final_features/qqp_train_features.pkl --custom_teacher ../teacher_preds/qqp_teacher_seed222.json


CUDA_VISIBLE_DEVICES=0 python get_final_features.py \
 --output_dir ../experiments_self_debias_qqp_seed777/bert_reweighted_sampled2K_teacher_seed666_annealed \
  --do_eval --gene_challenge --mode none\
 --seed 111 --which_bias qqp_hans_json --task qqp \
 --get_bert_output --shallow_feature_file shallow_features/shallow_paws.pkl \
 --final_feature final_features/paws_features.pkl --custom_teacher ../teacher_preds/qqp_teacher_seed222.json
