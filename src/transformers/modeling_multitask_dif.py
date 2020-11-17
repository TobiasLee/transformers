from .modeling_bert import *


class TaskSolver(nn.Module):
    def __init__(self, config, task_label_num=2, pooling='cls'):
        super(TaskSolver, self).__init__()
        self.output_layer_0 = nn.Linear(config.hidden_size, config.hidden_size)
        self.self_attn = BertSelfAttention(config)
        self.output_layer_1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.output_layer_2 = nn.Linear(config.hidden_size, task_label_num)
        self.pooling = pooling

    def forward(self, hidden, attention_mask):
        hidden = torch.tanh(self.output_layer_0(hidden))
        hidden = self.self_attn(hidden, attention_mask)[0]
        if self.pooling == "mean":
            hidden = torch.mean(hidden, dim=-1)
        elif self.pooling == "max":
            hidden = torch.max(hidden, dim=1)[0]
        elif self.pooling == "last":
            hidden = hidden[:, -1]
        else:  # default [CLS] pooling
            hidden = hidden[:, 0]
        output_1 = torch.tanh(self.output_layer_1(hidden))
        logits = self.output_layer_2(output_1)
        return logits


class DifficultyPredictor(nn.Module):
    def __init__(self, config, pooling='cls'):
        super(DifficultyPredictor, self).__init__()
        self.output_layer_0 = nn.Linear(config.hidden_size, config.hidden_size)
        self.self_attn = BertSelfAttention(config)
        self.output_layer_1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.output_layer_2 = nn.Linear(config.hidden_size, config.num_labels)  # difficulty task
        self.pooling = pooling

    def forward(self, hidden, attention_mask):
        hidden = torch.tanh(self.output_layer_0(hidden))
        hidden = self.self_attn(hidden, attention_mask)[0]
        if self.pooling == "mean":
            hidden = torch.mean(hidden, dim=1)
        elif self.pooling == "max":
            hidden = torch.max(hidden, dim=1)[0]
        elif self.pooling == "last":
            hidden = hidden[:, -1]
        else:  # default [CLS] pooling
            hidden = hidden[:, 0]
        output_1 = torch.tanh(self.output_layer_1(hidden))
        logits = self.output_layer_2(output_1)
        return logits


class BertForMultitaskClassification(BertPreTrainedModel):
    def __init__(self, config, task_label_num=2, task_pooling='cls', difficulty_pooling='mean'):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.task_num_labels = task_label_num
        # task
        self.task_classifier = TaskSolver(config, task_label_num=task_label_num, pooling=task_pooling)
        # difficulty classifier
        self.difficulty_classifier = DifficultyPredictor(config, pooling=difficulty_pooling)
        self.init_weights()

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
            task_labels=None,
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
        hidden_output = outputs[0]
        input_shape = input_ids.size()
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape, device)
        task_logits = self.task_classifier(hidden_output, extended_attention_mask)
        difficulty_logits = self.difficulty_classifier(hidden_output, extended_attention_mask)
        outputs = (difficulty_logits, task_logits)  # + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(difficulty_logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(difficulty_logits.view(-1, self.num_labels), labels.view(-1))

            if task_labels is not None:
                if self.task_num_labels == 1:
                    #  We are doing regression
                    loss_fct = MSELoss()
                    loss += loss_fct(task_logits.view(-1), task_labels.view(-1))
                else:
                    loss_fct = CrossEntropyLoss()
                    loss += loss_fct(task_logits.view(-1, self.task_num_labels), task_labels.view(-1))
                outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)
