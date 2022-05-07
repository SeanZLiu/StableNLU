for file_name in 111 222 333 444 555 666 679 777 888 999
do
CUDA_VISIBLE_DEVICES=0 python get_shallow_output_full_tasks.py \
 --output_dir ../experiments_shallow_fever/bert_base_sampled500_seed$file_name \
  --do_eval --gene_challenge --mode none\
 --seed 111 --which_bias fever_claim_only_balanced --task fever \
 --get_bert_output --shallow_feature_file shallow_features/shallow_symm.pkl
done
