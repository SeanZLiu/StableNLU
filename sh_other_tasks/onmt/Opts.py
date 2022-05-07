import argparse

parser = argparse.ArgumentParser()

def model_opts(parser):
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
    parser.add_argument("--dataset",
                        default='mnli',
                        type=str,
                        help="Which dataset is used for training.")
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
                        default=0.05,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--gpu_ls",
                        type=str,
                        default='0',
                        help="list of gpu for parallel training.")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")

    ## Our arguements
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
    parser.add_argument("--which_bias", choices=["hans", "hypo", "hans_json", "mix", "dam"], required=False)
    parser.add_argument("--custom_teacher", default=None)
    parser.add_argument("--custom_bias", default=None)
    parser.add_argument("--theta", type=float, default=0.1, help="for theta smoothed distillation loss")
    parser.add_argument("--add_bias_on_eval", action="store_true")
    
    ###
    parser.add_argument('--de_novo',
                        default=False,
                        action='store_true',
                        help='If the stable model is trained de novo.')
    
    parser.add_argument("--stable_model_path", default="/users4/ldu/stable_nli/emnlp2020-debiasing-unknown/experiments_self_debias_mnli/bert_product_sampled2K_baseline_seed111_annealed_selftrained_333", type=str,
                        help="Path for pertrained stable model.")
    parser.add_argument("--shallow_model_path", default="/users4/ldu/stable_nli/emnlp2020-debiasing-unknown/experiments_self_debias_mnli/shallow", type=str,
                        help="Path for shallow stable model.")
    parser.add_argument('--strategy_adv',
                        type=str, default='random_train',
                        help='How to obtain the adv samples to regularize the stable model.')
    parser.add_argument('--mode',
                        type=str, default='random',
                        help='How to utilize the shallow features.')
    parser.add_argument('--pret',
                        default=False, 
                        action='store_true',
                        help='How to utilize the shallow features.')
    parser.add_argument('--pret_features',
                        default=None, 
                        help='How to utilize the shallow features.')

    parser.add_argument('--use_superficial',
                        default=False,
                        action='store_true',
                        help='How to obtain the adv samples to regularize the stable model.')

    parser.add_argument('--shallow_feature_weight',
                        type=float, default=1,
                        help='A balance coeffieient controlling the porpotion of prediction loss and disentanle loss between F+ and F-.')

    parser.add_argument('--Lambda',
                        type=float, default=0.01,
                        help='A balance coeffieient controlling the porpotion of prediction loss and disentanle loss between F+ and F-.')
    parser.add_argument('--tau',
                        type=float, default=0.9,
                        help='A threshold beyond which we think the false features of two samples are similar enough to each other.')
    parser.add_argument('--tau_annealing',
                        default=False,
                        action='store_true',
                        help='If gradually increase the threshold tau.')

    parser.add_argument('--temperature',
                        default=0.1,
                        type=float,
                        help='The temperature constant for calculating InfoNCE.')
    parser.add_argument('--uplim_candi_list',
                        type=int, default=10,
                        help='The maximium sample number within the candidate list of a single sample.')
    parser.add_argument('--sample_size_c',
                        type=int, default=100000,
                        help='The sample size of the cirrculum datset (A).')
    parser.add_argument('--sample_size_e',
                        type=int, default=50000,
                        help='The sample size of the exam datset (B).')
    parser.add_argument('--thre_posi',
                        type=float, default=0.9,
                        help='The threhold measuring if a sample in datset A is correctly predicted with high confidence.')
    parser.add_argument('--thre_nega',
                        type=float, default=0.9,
                        help='The threhold measuring if a sample in datset B is wrongly predicted with high confidence.') 
                        
                                                     