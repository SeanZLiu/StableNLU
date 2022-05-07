CUDA_VISIBLE_DEVICES=0 python train_distill_bert.py \
 --output_dir ../experiments_shallow_mnli/bert_base_sampled2K_epoch5_seed135 \
 --do_train --do_eval --do_eval_on_train --mode none\
 --seed 135 --which_bias hans --debug --num_train_epochs 5 --debug_num 2000

CUDA_VISIBLE_DEVICES=0 python train_distill_bert.py \
 --output_dir ../experiments_shallow_mnli/bert_base_sampled2K_epoch5_seed246 \
 --do_train --do_eval --do_eval_on_train --mode none\
 --seed 246 --which_bias hans --debug --num_train_epochs 5 --debug_num 2000

CUDA_VISIBLE_DEVICES=0 python train_distill_bert.py \
 --output_dir ../experiments_shallow_mnli/bert_base_sampled2K_epoch5_seed357 \
 --do_train --do_eval --do_eval_on_train --mode none\
 --seed 357 --which_bias hans --debug --num_train_epochs 5 --debug_num 2000

CUDA_VISIBLE_DEVICES=0 python train_distill_bert.py \
 --output_dir ../experiments_shallow_mnli/bert_base_sampled2K_epoch5_seed468 \
 --do_train --do_eval --mode none\
 --seed 468 --which_bias hans --debug --num_train_epochs 5 --debug_num 2000

CUDA_VISIBLE_DEVICES=0 python train_distill_bert.py \
 --output_dir ../experiments_shallow_mnli/bert_base_sampled2K_epoch5_seed579 \
 --do_train --do_eval --mode none\
 --seed 579 --which_bias hans --debug --num_train_epochs 5 --debug_num 2000

CUDA_VISIBLE_DEVICES=0 python train_distill_bert.py \
 --output_dir ../experiments_shallow_mnli/bert_base_sampled2K_epoch5_seed680 \
 --do_train --do_eval --mode none\
 --seed 680 --which_bias hans --debug --num_train_epochs 5 --debug_num 2000

CUDA_VISIBLE_DEVICES=0 python train_distill_bert.py \
 --output_dir ../experiments_shallow_mnli/bert_base_sampled2K_epoch5_seed791 \
 --do_train --do_eval --mode none\
 --seed 791 --which_bias hans --debug --num_train_epochs 5 --debug_num 2000

CUDA_VISIBLE_DEVICES=0 python train_distill_bert.py \
 --output_dir ../experiments_shallow_mnli/bert_base_sampled2K_epoch5_seed802 \
 --do_train --do_eval --mode none\
 --seed 802 --which_bias hans --debug --num_train_epochs 5 --debug_num 2000

CUDA_VISIBLE_DEVICES=0 python train_distill_bert.py \
 --output_dir ../experiments_shallow_mnli/bert_base_sampled2K_epoch5_seed913 \
 --do_train --do_eval --mode none\
 --seed 913 --which_bias hans --debug --num_train_epochs 5 --debug_num 2000

