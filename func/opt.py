import argparse

parser = argparse.ArgumentParser()

def model_opts(parser):
    ## Required parameters

    parser.add_argument("--model_type", default="roberta", type=str,
                        help="bert,roberta,ernie")
    parser.add_argument("--get_bert_output", action='store_true',default=False)
    parser.add_argument("--shallow_feature_file", type=str, help="the pkl file to save the shallow feature.")
    parser.add_argument("--get_logits", action='store_true',default=False)
    parser.add_argument("--logits_file", type=str, help="the pkl file to save logits.")



    ## Required parameters
    parser.add_argument("--bert_model", default="bert-base-uncased", type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                             "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--cache_dir",
                        default="",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    # todo 需要确认除了bert的128意外，别的都是多少
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_eval_on_train",
                        action='store_true',
                        help="Whether to run eval on the train set.")
    parser.add_argument("--do_test",
                        action='store_true',
                        help="Whether to run test and create submission.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--seed",
                        default=None,
                        type=int,
                        help="Seed for randomized elements in the training")
    parser.add_argument("--eval_batch_size",
                        default=16,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")

    ## Our arguements
    parser.add_argument("--mode", choices=["none", "distill", "smoothed_distill", "smoothed_distill_annealed",
                                           "label_smoothing", "theta_smoothed_distill", "reweight_baseline",
                                           "smoothed_reweight_baseline", "permute_smoothed_distill",
                                           "bias_product_baseline", "learned_mixin_baseline",
                                           "reweight_by_teacher", "reweight_by_teacher_annealed",
                                           "bias_product_by_teacher", "bias_product_by_teacher_annealed",
                                           "focal_loss"])
    parser.add_argument("--penalty", type=float, default=0.03,
                        help="Penalty weight for the learn_mixin model")
    parser.add_argument("--focal_loss_gamma", type=float, default=1.0)
    parser.add_argument("--n_processes", type=int, default=4,
                        help="Processes to use for pre-processing")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug_num", type=int, default=2000)
    parser.add_argument("--sorted", action="store_true",
                        help='Sort the data so most batches have the same input length,'
                             ' makes things about 2x faster. Our experiments did not actually'
                             ' use this in the end (not sure if it makes a difference) so '
                             'its off by default.')
    parser.add_argument("--which_bias", choices=["hans", "hypo", "hans_json", "mix", "dam"], required=True)
    parser.add_argument("--custom_teacher", default=None)
    parser.add_argument("--custom_bias", default=None)
    parser.add_argument("--theta", type=float, default=0.1, help="for theta smoothed distillation loss")
    parser.add_argument("--add_bias_on_eval", action="store_true")
