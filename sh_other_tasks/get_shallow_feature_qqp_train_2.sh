for file_name in 666 679 777 888 999
do
CUDA_VISIBLE_DEVICES=1 python get_shallow_output_full_tasks.py \
 --output_dir ../experiments_shallow_qqp/bert_base_sampled500_seed$file_name \
  --do_eval --do_eval_on_train --mode none\
 --seed 111 --which_bias qqp_hans_json --task qqp \
   --get_bert_output --shallow_feature_file shallow_features/shallow_qqp_train1.pkl --custom_teacher ../teacher_preds/qqp_teacher_seed222.json
done
