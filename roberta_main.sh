CUDA_VISIBLE_DEVICES=0 python train_roberta_distill.py \
  --train_with_feature --model_type roberta --output_dir ../experiments_self_debias_mnli_seed111_different_shallow/roberta_main/roberta_seed791_bert_3 \
  --do_train --do_eval --mode none --custom_teacher ../teacher_preds/mnli_trained_on_sample2K_seed111.json\
 --seed 791 --which_bias hans --task mnli --shallow_model_num 3 --shallow_feature_file shallow_features/shallow_mnli_train_5epoch.pkl
