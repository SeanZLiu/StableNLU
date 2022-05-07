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
