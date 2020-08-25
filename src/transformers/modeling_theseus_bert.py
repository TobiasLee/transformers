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

    def forward(self, hidden_states):
        # Pooler
        pooler_output = self.pooler(hidden_states)
        # "return" pooler_output

        pooled_output = self.dropout(pooler_output)
        logits = self.classifier(pooled_output)

        return logits


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
    def __init__(self, config, scc_n_layer=6, switch_pattern=0, num_parts=6, switch_mode=False, n_action_space=3):
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
        # self.base_early_exits = nn.ModuleList([EarlyClassifier(config) for _ in range(scc_n_layer)])
        # self.large_early_exits = nn.ModuleList([EarlyClassifier(config) for _ in range(config.num_hidden_layers)])
        self.early_classifiers = nn.ModuleList([EarlyClassifier(config) for _ in range(self.scc_n_layer)])
        self.switch_mode = switch_mode
        self.agent = SwitchAgent(config, n_action_space=n_action_space)

    def set_replacing_rate(self, replacing_rate):
        if not 0 < replacing_rate <= 1:
            raise Exception('Replace rate must be in the range (0, 1]!')
        self.bernoulli = Bernoulli(torch.tensor([replacing_rate]))

    def init_highway_pooler(self, pooler):
        # 实际上在 copy 最后一层 pooler
        loaded_model = pooler.state_dict()
        for highway in self.early_classifiers:
            for name, param in highway.pooler.state_dict().items():
                param.copy_(loaded_model[name])
        # for highway in self.base_early_exits:
        #     for name, param in highway.pooler.state_dict().items():
        #         param.copy_(loaded_model[name])
        #
        # for highway in self.large_early_exits:
        #     for name, param in highway.pooler.state_dict().items():
        #         param.copy_(loaded_model[name])

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
            bsz = hidden_states.size()[0]
            device = hidden_states.device
            left_idx = torch.arange(bsz, device=device)
            large_interval = self.prd_n_layer // self.num_parts
            base_interval = self.scc_n_layer // self.num_parts
            # training with a switch agent
            action_probs = []
            actions = []
            internal_classifier_logits = []

            if self.training:  #
                internal_base_hidden, internal_large_hidden = hidden_states, hidden_states

                for i in range(self.num_parts):
                    # internal  logit for training early exits
                    internal_base_logit = self.early_classifiers[i](internal_base_hidden)
                    internal_large_logit = self.early_classifiers[i](internal_large_hidden)
                    internal_classifier_logits.append(internal_base_logit)
                    internal_classifier_logits.append(internal_large_logit)
                    internal_base_hidden, _ = _run_sub_blocks(internal_base_hidden,
                                                              self.scc_layer[
                                                              i * base_interval:i * base_interval + base_interval],
                                                              left_idx)
                    internal_large_hidden, _ = _run_sub_blocks(internal_large_hidden,
                                                               self.layer[
                                                               i * large_interval:i * large_interval + large_interval],
                                                               left_idx)

            early_exit_pairs = []
            for i in range(self.num_parts):
                if len(hidden_states) == 0 :
                    break 
                action_prob = self.agent(hidden_states)
                padded_prob = torch.ones((bsz, self.agent.action_classifier.out_features), device=device)
                padded_prob[left_idx] = action_prob
                action_probs.append(padded_prob)
                # policy gradient
                if self.training:
                    m = Categorical(action_prob)
                    action = m.sample()
                else:  # during evaluation, we do not sample but using the argmax for path selection
                    action = torch.argmax(action_prob, dim=-1)
                padded_action = torch.zeros(size=(bsz,), device=device, dtype=torch.long)
                padded_action[left_idx] = action
                actions.append(padded_action)

                exit_idx = left_idx[action == 0]  # using 0 for current code
                if len(exit_idx) > 0:
                    exited_logit = self.early_classifiers[i](hidden_states)[action == 0]
                    early_exit_pairs.append((exited_logit, exit_idx))

                #  to implement acceleration, exited examples are not supposed to continue the forward loop

                base_idx = left_idx[action == 1]
                large_idx = left_idx[action == 2]
                base_input = hidden_states[action == 1]
                large_input = hidden_states[action == 2]

                if len(base_input) > 0:
                    base_hiddens, base_outputs = _run_sub_blocks(base_input,
                                                                 self.scc_layer[
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
                sorted_idx, order = torch.sort(torch.cat((base_idx, large_idx), dim=0))
                left_idx = sorted_idx
                hidden_states = hidden_states[order]  # order left hidden it back

                if self.output_hidden_states:
                    all_hidden_states = all_hidden_states + (
                        (base_hiddens, large_hiddens),)  # emit for the first hidden states?
                if self.output_attentions:
                    all_attentions = all_attentions + ((base_outputs[1], large_outputs[1]),)

            outputs = (hidden_states,)
            # stack results for fp 16
            stacked_probs = torch.cat([p.unsqueeze(0) for p in action_probs], dim=0)  # num_parts, bsz, actio_space
            stacked_action = torch.cat([a.unsqueeze(0) for a in actions])  # num_parts, bsz,
            if len(early_exit_pairs) > 0:
                early_exit_logit = torch.cat([p[0] for p in early_exit_pairs], dim=0)  # num_exited,  num_labels
                early_exit_idx = torch.cat([p[1] for p in early_exit_pairs], dim=0)  # num_exited,
            else:
                early_exit_logit = None
                early_exit_idx = None

            if len(internal_classifier_logits) > 0:
                stacked_internal_classifier_logits = torch.cat(
                    [logit.unsqueeze(0) for logit in internal_classifier_logits],
                    dim=0)  # num_parts, bsz, num_labels
            else:
                stacked_internal_classifier_logits = None

            outputs = outputs + (left_idx,
                                 stacked_probs,
                                 stacked_action,
                                 early_exit_logit,
                                 early_exit_idx,
                                 stacked_internal_classifier_logits,
                                 )  # action_probs for computing loss
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
        left_idx = None
        early_exit_idx = None
        early_exit_logit = None
        internal_classifier_logits = None
        if self.bert.encoder.switch_mode:
            left_idx = outputs[2]
            action_probs = outputs[3]
            actions = outputs[4]
            early_exit_logit = outputs[5]
            early_exit_idx = outputs[6]
            internal_classifier_logits = outputs[7]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        # early_exit_logit = torch.cat([p[0] for p in early_exit_pairs], dim=0)
        # early_exit_idx = torch.cat([p[1] for p in early_exit_pairs], dim=0)
        if early_exit_idx is not None and len(early_exit_idx) > 0:  # if early exit happens, re-order it back
            total_idx = torch.cat([early_exit_idx, left_idx], dim=0)
            _, order = torch.sort(total_idx)
            logits = torch.cat([early_exit_logit, logits], dim=0)[order]

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
                internal_loss = None
                weights = 0.0
                # internal classifier loss
                if self.training:
                    for i, logits in enumerate(internal_classifier_logits):
                        if internal_loss is None:
                            internal_loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                        else:
                            internal_loss += (i + 1) * loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                        weights += i + 1
                internal_loss = internal_loss / weights if internal_loss is not None else 0.0

                bsz = logits.size()[0]
                final_decision_prob = torch.ones((bsz,), device=input_ids.device)
                paths = []
                path_penalty = torch.zeros((bsz,), device=input_ids.device)
                for path_prob, action in zip(action_probs, actions):
                    selected_path = action.unsqueeze(1)  # bsz, 1
                    prob = torch.gather(path_prob, dim=-1, index=selected_path).squeeze()  # bsz
                    final_decision_prob *= prob
                    path_penalty += action  #
                    paths.append(action.unsqueeze(1))

                    # padded_prob = torch.ones((bsz,), device=input_ids.device)
                    # padded_prob[action_idx] = prob
                    # final_decision_prob *= padded_prob  # final prob
                    # padded_path = torch.zeros((bsz,), device=input_ids.device, dtype=torch.long)
                    # padded_path[action_idx] = action
                    # path_penalty += padded_path  # add large block prob as penalty
                    # paths.append(padded_path.unsqueeze(1))

                if not self.training:
                    paths = torch.cat(paths, dim=-1)  # bsz, num_parts
                    # we can add an expected saving computation here
                    all_large = torch.ones_like(paths) * 2
                    print("Layer ratio: %.3f%%" % (
                            (torch.sum(paths) / torch.sum(all_large, dtype=torch.float)).item() * 100))
                    print(paths[:4])  # sample for some path

                if self.num_labels != 1:
                    entropy_reward_fct = CrossEntropyLoss(reduction='none')
                    reward = - entropy_reward_fct(logits.view(-1, self.num_labels), labels.view(-1))
                else:
                    mse_reward_fct = MSELoss(reduction='none')
                    reward = - mse_reward_fct(logits.view(-1), labels.view(-1))

                reward -= self.path_penalty_ratio * path_penalty
                reward = torch.mean(reward *
                                    torch.log(final_decision_prob + 1e-9))  # sum over bsz
                loss = loss + internal_loss - reward  # minus reward + penalty
                del paths
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
