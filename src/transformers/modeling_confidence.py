from .modeling_bert import *
import torch.nn.functional as F


class BertConfidenceAwareClassification(BertPreTrainedModel):
    def __init__(self, config, task_pooling='cls', difficulty_pooling='mean'):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # task
        self.task_classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.init_weights()
        self.ranking_loss = nn.MarginRankingLoss(margin=0.0)
        self.margin_loss = nn.MarginRankingLoss(margin=0.2)

    @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING.format("(batch_size, sequence_length)"))
    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            difficulty_labels=None,
            mlp_mask=None
    ):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        task_logits = self.task_classifier(pooled_output)
        outputs = (task_logits, task_logits,)  # + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(task_logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(task_logits.view(-1, self.num_labels), labels.view(-1))
            # ---------------------------------- #
            if difficulty_labels is not None:
                conf = F.softmax(task_logits, dim=-1)  # bsz, num_label
                confidence, _ = conf.max(dim=-1)
                loss += self.confidence_loss(difficulty_labels, confidence)  # pair loss between 0 and other examples
                loss += self.pair_loss(difficulty_labels, confidence, dif1=1, dif2=2)
                outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)

    def _get_target_margin(self, dif1, dif2):
        geq = torch.where(dif1 >= dif2, torch.ones_like(dif1), torch.zeros_like(dif1))
        less = torch.where(dif1 < dif2, -1 * torch.ones_like(dif2), torch.zeros_like(dif2))
        margin = torch.abs(dif1 - dif2)
        target = geq + less
        return target, margin

    def pair_loss(self, difficulty_labels, confidence, dif1=0, dif2=1):
        # we first try to adjust difficulty = 0  and difficulty = 1
        easy_idx = (difficulty_labels == dif1)
        hard_idx = (difficulty_labels == dif2)
        easy_conf = confidence[easy_idx]
        hard_conf = confidence[hard_idx]
        if len(easy_conf) == 0 or len(hard_conf) == 0:
            return 0.0
        uniform = torch.ones_like(hard_conf) / len(hard_conf)
        sampled_hard_idx = torch.multinomial(uniform, num_samples=len(easy_conf), replacement=True)
        rank_input1 = easy_conf
        rank_input2 = hard_conf[sampled_hard_idx]
        diff_label1 = 1.0 / (1.0 + difficulty_labels[easy_idx])  # 1.0
        diff_label2 = 1.0 / (1.0 + difficulty_labels[hard_idx][sampled_hard_idx])  # 0.5
        target, _ = self._get_target_margin(diff_label1, diff_label2)
        confidence_loss = self.margin_loss(rank_input1, rank_input2, target)
        return confidence_loss

    def confidence_loss(self, difficulty_labels, confidence):
        difficulty_idx = (difficulty_labels != 0)  # index for dif examples
        assert self.num_labels != 1, "We do not support regression task for now!"
        # pair each hard examples with a simple examples
        hard_dif_labels = difficulty_labels[difficulty_idx]
        easy_dif_labels = difficulty_labels[~difficulty_idx]
        if len(hard_dif_labels) == 0 or len(easy_dif_labels) == 0:  # if there is no hard/easy examples
            return 0.0
        hard_conf = confidence[difficulty_idx]
        easy_conf = confidence[~difficulty_idx]

        uniform = torch.ones_like(hard_conf) / len(hard_dif_labels)
        sampled_hard_idx = torch.multinomial(uniform, num_samples=len(easy_dif_labels), replacement=True)
        uniform = torch.ones_like(easy_conf) / len(easy_dif_labels)
        sampled_easy_idx = torch.multinomial(uniform, num_samples=len(hard_dif_labels), replacement=True)

        rank_input1 = torch.cat([hard_conf, easy_conf], dim=0)
        rank_input2 = torch.cat([easy_conf[sampled_easy_idx], hard_conf[sampled_hard_idx]], dim=0)
        # dif_label1  dif_label2
        # confidence1 confidence2
        dif_for_easy_examples = difficulty_labels[difficulty_idx][sampled_hard_idx]
        dif_for_hard_examples = difficulty_labels[~difficulty_idx][sampled_easy_idx]

        diff_label1 = torch.cat([hard_dif_labels, easy_dif_labels], dim=0)
        diff_label2 = torch.cat([dif_for_hard_examples, dif_for_easy_examples], dim=0)
        # normalization to proper range
        diff_label1 = 1.0 / (diff_label1 + 1.0)
        diff_label2 = 1.0 / (diff_label2 + 1.0)

        target, margin = self._get_target_margin(diff_label1, diff_label2)
        # confidence aware learning
        # rank_input2 = rank_input2 + torch.true_divide(margin, target)
        # confidence_loss = self.ranking_loss(rank_input1, rank_input2, target)

        # simple margin loss
        confidence_loss = self.margin_loss(rank_input1, rank_input2, target)
        return confidence_loss
