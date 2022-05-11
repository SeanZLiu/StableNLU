
for shallow_num in 1 3 5 7 9
do
CUDA_VISIBLE_DEVICES=0 python train_bert_features.py \
 --linear  --output_dir linear_net/linear_seed357_ernie_shallow$shallow_num \
  --do_train  --mode none --custom_teacher ../teacher_preds/mnli_trained_on_sample2K_seed111.json\
 --seed 357 --which_bias hans --task mnli --shallow_model_num $shallow_num --shallow_feature_file shallow_features/shallow_mnli_train_5epoch_ernie.pkl
done
