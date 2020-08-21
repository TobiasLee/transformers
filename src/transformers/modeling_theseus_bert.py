"""PyTorch BERT-of-Theseus model. """

from __future__ import absolute_import, division, print_function, unicode_literals

import logging

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from torch.distributions.bernoulli import Bernoulli
from torch.distributions import Categorical
from transformers.modeling_utils import PreTrainedModel
from transformers.configuration_bert import BertConfig
from transformers.modeling_bert import load_tf_weights_in_bert, BertLayerNorm, BertEmbeddings, BertLayer, BertPooler

logger = logging.getLogger(__name__)


class EarlyClassifier(nn.Module):
    r"""A module to provide a shortcut
    from
    the output of one non-final BertLayer in BertEncoder
    to
    cross-entropy computation in BertForSequenceClassification
    """

    def __init__(self, config):
        super(EarlyClassifier, self).__init__()
        self.pooler = BertPooler(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, encoder_outputs):
        # Pooler
        pooler_input = encoder_outputs[0]
        pooler_output = self.pooler(pooler_input)
        # "return" pooler_output

        # BertModel
        bmodel_output = (pooler_input, pooler_output) + encoder_outputs[1:]
        # "return" bodel_output

        # Dropout and classification
        pooled_output = bmodel_output[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        return logits, pooled_output


class SwitchAgent(nn.Module):
    def __init__(self, config, n_action_space=2):
        super(SwitchAgent, self).__init__()
        self.pooler = BertPooler(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.action_classifier = nn.Linear(config.hidden_size, n_action_space)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, hidden_states):
        # we pooling the hidden states using the first CLS representation
        pooler_output = self.pooler(hidden_states)
        pooler_output = self.dropout(pooler_output)
        action_logit = self.action_classifier(pooler_output)
        action_prob = self.softmax(action_logit)
        return action_prob


class BertEncoder(nn.Module):
    def __init__(self, config, scc_n_layer=6, switch_pattern=0, num_parts=6, switch_mode=False, n_action_space=2):
        super(BertEncoder, self).__init__()
        self.prd_n_layer = config.num_hidden_layers
        self.scc_n_layer = scc_n_layer
        assert self.prd_n_layer % self.scc_n_layer == 0
        self.compress_ratio = self.prd_n_layer // self.scc_n_layer
        self.bernoulli = None
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(self.prd_n_layer)])
        self.scc_layer = nn.ModuleList([BertLayer(config) for _ in range(self.scc_n_layer)])
        self.switch_pattern = switch_pattern
        self.num_parts = num_parts
        self.base_early_exits = nn.ModuleList([EarlyClassifier(config) for _ in range(scc_n_layer)])
        self.large_early_exits = nn.ModuleList([EarlyClassifier(config) for _ in range(config.num_hidden_layers)])
        self.switch_mode = switch_mode
        self.agent = SwitchAgent(config, n_action_space=n_action_space)

    def set_replacing_rate(self, replacing_rate):
        if not 0 < replacing_rate <= 1:
            raise Exception('Replace rate must be in the range (0, 1]!')
        self.bernoulli = Bernoulli(torch.tensor([replacing_rate]))

    def init_highway_pooler(self, pooler):
        # 实际上在 copy 最后一层 pooler
        loaded_model = pooler.state_dict()
        for highway in self.base_early_exits:
            for name, param in highway.pooler.state_dict().items():
                param.copy_(loaded_model[name])

        for highway in self.large_early_exits:
            for name, param in highway.pooler.state_dict().items():
                param.copy_(loaded_model[name])

    def forward(self, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None,
                encoder_attention_mask=None):
        all_hidden_states = ()
        all_attentions = ()

        def _run_sub_blocks(layer_hidden_states, layers, idx):
            for i, layer in enumerate(layers):
                # print('layer :%d' %i )
                # print(attention_mask[idx].size())
                layer_output = layer(layer_hidden_states, attention_mask[idx], None, encoder_hidden_states,
                                     encoder_attention_mask[idx] if encoder_attention_mask is not None else None)
                layer_hidden_states = layer_output[0]
            return layer_hidden_states, layer_output

        if self.training and not self.switch_mode:
            inference_layers = []
            for i in range(self.scc_n_layer):
                if self.bernoulli.sample() == 1:  # REPLACE
                    inference_layers.append(self.scc_layer[i])
                else:  # KEEP the original
                    for offset in range(self.compress_ratio):
                        inference_layers.append(self.layer[i * self.compress_ratio + offset])

        elif self.switch_mode:
            # training with a switch agent
            # hidden_states
            bsz = hidden_states.size()[0]
            idx = torch.arange(bsz)
            classifier_indicator = None
            large_interval = self.prd_n_layer // self.num_parts
            base_interval = self.scc_n_layer // self.num_parts
            action_probs = []
            actions = []
            for i in range(self.num_parts):
                # print('num_parts:', i)
                action_prob = self.agent(hidden_states)
                action_probs.append(action_prob)
                # policy gradient
                m = Categorical(action_prob)
                action = m.sample()
                actions.append(action)
                #$ torch.argmax(action_prob, dim=-1)  # make action based on current hidden state, [bsz, ]
                base_idx = idx[action == 0]
                large_idx = idx[action == 1]
                base_input = hidden_states[base_idx]
                large_input = hidden_states[large_idx]
                # print('base: ', base_idx)
                # print('large: ', large_idx)
                if len(base_input) > 0:
                    base_hiddens, base_outputs = _run_sub_blocks(base_input, self.scc_layer[
                                                                             i * base_interval:i * base_interval + base_interval],
                                                                 base_idx)
                if len(large_input) > 0:
                    large_hiddens, large_outputs = _run_sub_blocks(large_input, self.layer[
                                                                                i * large_interval:i * large_interval + large_interval],
                                                                   large_idx)
                if len(base_input) == 0:
                    hidden_states = large_hiddens
                elif len(large_input) == 0:
                    hidden_states = base_hiddens
                else:
                    hidden_states = torch.cat((base_hiddens, large_hiddens), dim=0)

                _, order = torch.sort(torch.cat((base_idx, large_idx), dim=0))
                hidden_states = hidden_states[order]  # order it back
                if self.output_hidden_states:
                    all_hidden_states = all_hidden_states + (
                        (base_hiddens, large_hiddens),)  # emit for the first hidden states?
                if self.output_attentions:
                    all_attentions = all_attentions + ((base_outputs[1], large_outputs[1]),)

            outputs = (hidden_states,)
            outputs = outputs + (action_probs, actions, )  # action_probs for computing loss
            if self.output_hidden_states:
                outputs = outputs + (all_hidden_states,)
            if self.output_attentions:
                outputs = outputs + (all_attentions,)
            return outputs  # last-layer hidden state, action_probs, (all hidden states), (all attentions)

        else:  # inference with compressed model
            if self.switch_pattern == 0:  # default setting
                inference_layers = self.scc_layer
            elif self.switch_pattern > 0:
                assert self.switch_pattern < 2 ** self.num_parts, "switch pattern idx should ranges from 0 to 2^num_parts -1"
                inference_layers = []
                pattern = self.switch_pattern
                large_layers, base_layers = [], []
                large_interval = self.prd_n_layer // self.num_parts  #
                base_interval = self.scc_n_layer // self.num_parts
                for i in range(self.num_parts):
                    large_layers.append(self.layer[i * large_interval:i * large_interval + large_interval])
                    base_layers.append(self.scc_layer[i * base_interval:i * base_interval + base_interval])

                for i in range(self.num_parts):  # indeed, it is a six switch model
                    if pattern % 2 == 1:  # large:
                        inference_layers.extend(large_layers[i])
                        # for offset in range(self.compress_ratio):
                        #     inference_layers.append(self.layer[i * self.compress_ratio + offset])
                    else:
                        inference_layers.extend(base_layers[i])
                    pattern //= 2

        for i, layer_module in enumerate(inference_layers):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(hidden_states, attention_mask, head_mask[i], encoder_hidden_states,
                                         encoder_attention_mask)
            hidden_states = layer_outputs[0]

            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs  # last-layer hidden state, (all hidden states), (all attentions)


class ConstantReplacementScheduler:
    def __init__(self, bert_encoder: BertEncoder, replacing_rate, replacing_steps=None):
        self.bert_encoder = bert_encoder
        self.replacing_rate = replacing_rate
        self.replacing_steps = replacing_steps
        self.step_counter = 0
        self.bert_encoder.set_replacing_rate(replacing_rate)

    def step(self):
        self.step_counter += 1
        if self.replacing_steps is None or self.replacing_rate == 1.0:
            return self.replacing_rate
        else:
            if self.step_counter >= self.replacing_steps:
                self.bert_encoder.set_replacing_rate(1.0)
                self.replacing_rate = 1.0
            return self.replacing_rate


class LinearReplacementScheduler:
    def __init__(self, bert_encoder: BertEncoder, base_replacing_rate, k):
        self.bert_encoder = bert_encoder
        self.base_replacing_rate = base_replacing_rate
        self.step_counter = 0
        self.k = k
        self.bert_encoder.set_replacing_rate(base_replacing_rate)

    def step(self):
        self.step_counter += 1
        current_replacing_rate = min(self.k * self.step_counter + self.base_replacing_rate, 1.0)
        self.bert_encoder.set_replacing_rate(current_replacing_rate)
        return current_replacing_rate


class BertPreTrainedModel(PreTrainedModel):
    config_class = BertConfig
    load_tf_weights = load_tf_weights_in_bert
    base_model_prefix = "bert"

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


class BertModel(BertPreTrainedModel):
    def __init__(self, config, switch_pattern=0):
        super(BertModel, self).__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config, switch_pattern=switch_pattern)
        self.pooler = BertPooler(config)

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def init_highway_pooler(self):
        self.encoder.init_highway_pooler(self.pooler)

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None,
                head_mask=None, inputs_embeds=None, encoder_hidden_states=None, encoder_attention_mask=None):
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastabe to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(input_ids=input_ids, position_ids=position_ids,
                                           token_type_ids=token_type_ids, inputs_embeds=inputs_embeds)
        encoder_outputs = self.encoder(embedding_output,
                                       attention_mask=extended_attention_mask,
                                       head_mask=head_mask,
                                       encoder_hidden_states=encoder_hidden_states,
                                       encoder_attention_mask=encoder_extended_attention_mask)
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        outputs = (sequence_output, pooled_output,) + encoder_outputs[
                                                      1:]  # add hidden_states and attentions if they are here
        return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)


class BertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)

        self.init_weights()
        self.path_penalty_ratio = 0.0

    def set_switch_pattern(self, switch_pattern):
        self.bert.encoder.switch_pattern = switch_pattern

    def set_switch_mode(self, switch_mode):
        self.bert.encoder.switch_mode = switch_mode

    def set_path_penalty(self, penalty_ratio):
        self.path_penalty_ratio = penalty_ratio

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, inputs_embeds=None, labels=None):

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds)

        pooled_output = outputs[1]
        action_probs = None
        actions = None
        if self.bert.encoder.switch_mode:
            action_probs = outputs[2]
            actions = outputs[3]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            if action_probs is not None:
                bsz = logits.size()[0]
                final_decision_prob = torch.ones((bsz,), device=input_ids.device)
                paths = []
                path_penalty = torch.zeros((bsz,), device=input_ids.device)
                for path_prob, action in zip(action_probs, actions):
                    selected_path = action.unsqueeze(1)
                    prob = torch.gather(path_prob, dim=-1, index=selected_path)
                    final_decision_prob *= prob  # final prob
                    path_penalty += (action + 1 ) # add large block prob as penalty
                    paths.append(selected_path)
                if not self.training:
                    paths = torch.cat(paths, dim=-1)  # bsz, num_parts
                    print(paths[:4])  # sample for some path
                if self.num_labels != 1:
                    entropy_reward_fct = CrossEntropyLoss(reduction='none')
                    reward = - entropy_reward_fct(logits.view(-1, self.num_labels), labels.view(-1))
                else:
                    mse_reward_fct = MSELoss(reduction='none')
                    reward = - mse_reward_fct(logits.view(-1), labels.view(-1))

                reward -= self.path_penalty_ratio * path_penalty
                classification_reward = torch.sum(reward *
                                                  torch.log(final_decision_prob + 1e-9))  # sum over bsz
                loss = loss - classification_reward + path_penalty  # minus reward + penalty

            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


class BertForMultipleChoice(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForMultipleChoice, self).__init__(config)

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)

        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, inputs_embeds=None, labels=None):
        num_choices = input_ids.shape[1]

        input_ids = input_ids.view(-1, input_ids.size(-1))
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds)

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)

        outputs = (reshaped_logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)
            outputs = (loss,) + outputs

        return outputs  # (loss), reshaped_logits, (hidden_states), (attentions)


class BertForTokenClassification(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForTokenClassification, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, inputs_embeds=None, labels=None):

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds)

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), scores, (hidden_states), (attentions)


class BertForQuestionAnswering(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForQuestionAnswering, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None,
                start_positions=None, end_positions=None):

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds)

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        outputs = (start_logits, end_logits,) + outputs[2:]
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            outputs = (total_loss,) + outputs

        return outputs  # (loss), start_logits, end_logits, (hidden_states), (attentions)
