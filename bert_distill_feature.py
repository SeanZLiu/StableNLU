from pytorch_pretrained_bert.modeling import BertModel, BertPreTrainedModel
from torch import nn
from torch.nn import functional as F
from torch.nn import CrossEntropyLoss
import torch
from clf_distill_loss_functions import ClfDistillLossFunction


class LinearNet(nn.Module):
    def __init__(self, input_size=768,num_labels=3):
        super(LinearNet, self).__init__()
        self.net = nn.Linear(input_size, num_labels)

    def forward(self, labels=None, bias_features=None):
        logits = self.net(bias_features)
        if labels is None:
            return logits
        loss = F.cross_entropy(logits, labels)
        return logits, loss


class BertDistill(BertPreTrainedModel):
    """Pre-trained BERT model that uses our loss functions"""

    def __init__(self, config, num_labels, loss_fn: ClfDistillLossFunction):
        super(BertDistill, self).__init__(config)
        self.num_labels = num_labels
        self.loss_fn = loss_fn
        self.bert = BertModel.from_pretrained("/users5/kxiong/huggingface_transformers/bert-base-uncased")
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        # self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,
                labels=None, bias=None, teacher_probs=None, bias_features=None):
        _, pooled_output = self.bert(
            input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        #logits = self.classifier(self.dropout(pooled_output))
        # todo 添加取均值后的表面特征
        #bias_features = torch.Tensor(bias_features)
        logits = self.classifier(self.dropout(pooled_output + bias_features))
        # todo

        if labels is None:
            return logits
        loss = F.cross_entropy(logits, labels)
        #loss = self.loss_fn.forward(pooled_output, logits, bias, teacher_probs, labels)
        return logits, loss

    def get_bert_output(self, input_ids, token_type_ids=None, attention_mask=None,
                labels=None, bias=None, teacher_probs=None):
        _, pooled_output = self.bert(
            input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        return pooled_output


    def forward_and_log(self, input_ids, token_type_ids=None, attention_mask=None,
                labels=None, bias=None, teacher_probs=None):
        _, pooled_output = self.bert(
            input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        logits = self.classifier(self.dropout(pooled_output))
        if labels is None:
            return logits
        loss = self.loss_fn.forward(pooled_output, logits, bias, teacher_probs, labels)

        cel_fct = CrossEntropyLoss(reduction="none")
        indv_losses = cel_fct(logits, labels).detach()
        return logits, loss, indv_losses
    
    def forward_analyze(self, input_ids, token_type_ids=None, attention_mask=None,
                labels=None, bias=None, teacher_probs=None):
        _, pooled_output = self.bert(
            input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        logits = self.classifier(self.dropout(pooled_output))

        if labels is None and not self.training:
            return logits, pooled_output
        else:
            raise Exception("should be called during eval and "
                            "labels should be none")
