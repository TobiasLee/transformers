"""PyTorch BERT-of-Theseus model. """

from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from typing import Tuple

import torch
import torch.autograd as autograd
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from torch.distributions.bernoulli import Bernoulli
from torch.distributions import Categorical
from transformers.modeling_utils import PreTrainedModel
from transformers.configuration_bert import BertConfig
from transformers.modeling_bert import load_tf_weights_in_bert, BertLayerNorm, BertEmbeddings, BertLayer, BertPooler

import random

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


class RNNSwitchAgent(nn.Module):
    def __init__(self, config, hidden_size, n_action_space=2, rnn_type='lstm'):
        super(RNNSwitchAgent, self).__init__()
        self.pooler = BertPooler(config)
        if rnn_type == "lstm":
            self.rnn = nn.LSTM(self.pooler.dense.out_features, hidden_size)
        else:
            raise ValueError("current not supported other type rnn")
        self.hidden = None
        self.linear = nn.Linear(hidden_size, n_action_space)
        self.softmax = nn.Softmax(dim=-1)
        self.hidden_size = hidden_size

    def reset_hidden(self, batch_size, device, dtype, requires_grad=True):
        return (torch.zeros((1, batch_size, self.hidden_size), device=device, requires_grad=requires_grad, dtype=dtype),
                torch.zeros((1, batch_size, self.hidden_size), device=device, requires_grad=requires_grad, dtype=dtype))
        # self.hidden = None

    def forward(self, hidden_states, rnn_hidden):
        pooler_output = self.pooler(hidden_states)
        bsz = pooler_output.size(0)
        self.rnn.flatten_parameters()
        # rnn_hidden = None if self.hidden is None else (self.hidden[0][:, left_idx], self.hidden[1][:, left_idx])
        out, new_hiddens = self.rnn(
            pooler_output.view(1, bsz, -1), rnn_hidden
        )
        # if self.hidden is None:  # first entry
        #    self.hidden = new_hiddens
        # else:
        # update hiddens
        #    self.hidden[0][:, left_idx], self.hidden[1][:, left_idx] = new_hiddens[0], new_hiddens[1]
        out = out.squeeze(0)
        action_logit = self.linear(out)
        action_prob = self.softmax(action_logit)
        return action_prob, new_hiddens


class MeanPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        mean_token_tensor = torch.mean(hidden_states, dim=1)
        pooled_output = self.dense(mean_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


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
    def __init__(self, config, scc_n_layer=6, switch_pattern=0, num_parts=6,
                 train_agent=False, n_action_space=3,
                 train_early_exit=False,
                 early_exit_idx=-1,
                 only_large_and_exit=False,
                 bound_alpha=-1.0):
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
        self.train_agent = train_agent
        self.agent = SwitchAgent(config, n_action_space=n_action_space)  # n_action_space)
        self.simple_agent = SwitchAgent(config, n_action_space=2)
        self.config = config
        self.train_early_exit = train_early_exit
        self.early_exit_idx = early_exit_idx
        self.only_large_and_exit = only_large_and_exit
        self.bound_alpha = bound_alpha
        self.cl_idx = -1
        self.rnn_agent = RNNSwitchAgent(config, hidden_size=32, n_action_space=n_action_space,
                                        rnn_type="lstm")

    def set_replacing_rate(self, replacing_rate):
        if not 0 < replacing_rate <= 1:
            raise Exception('Replace rate must be in the range (0, 1]!')
        self.bernoulli = Bernoulli(torch.tensor([replacing_rate]))

    def set_bound_alpha(self, alpha):
        self.bound_alpha = alpha

    def set_cl_idx(self, index):
        """how many num blocks is defined to use large blocks, ranges from [1, num_parts]"""
        self.cl_idx = index

    def init_agent_pooler(self, pooler):
        loaded_model = pooler.state_dict()
        for name, param in self.agent.pooler.state_dict().items():
            param.copy_(loaded_model[name])
        for name, param in self.rnn_agent.pooler.state_dict().items():
            param.copy_(loaded_model[name])

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

        if self.training and not self.train_agent and not self.train_early_exit:  # normal theseus training
            inference_layers = []
            for i in range(self.scc_n_layer):
                if self.bernoulli.sample() == 1:  # REPLACE
                    inference_layers.append(self.scc_layer[i])
                else:  # KEEP the original
                    for offset in range(self.compress_ratio):
                        inference_layers.append(self.layer[i * self.compress_ratio + offset])

        elif self.train_early_exit:
            # internal_base_hidden, internal_large_hidden = hidden_states, hidden_states
            bsz = hidden_states.size()[0]
            device = hidden_states.device
            left_idx = torch.arange(bsz, device=device)
            large_interval = self.prd_n_layer // self.num_parts  #
            base_interval = self.scc_n_layer // self.num_parts

            # pattern = 0  # random.choice([i for i in range(0, 2 ** self.num_parts)])
            # all_early_logits = ()

            def _run(pattern_idx):
                internal_hidden = hidden_states
                early_exits_logits = ()
                for i in range(self.num_parts):  # indeed, it is a six switch model
                    internal_logit = self.early_classifiers[i](internal_hidden)
                    early_exits_logits = early_exits_logits + (internal_logit,)
                    if pattern_idx % 2 == 1:
                        internal_hidden, _ = _run_sub_blocks(internal_hidden,
                                                             self.layer[
                                                             i * large_interval:i * large_interval + large_interval],
                                                             left_idx)

                    else:
                        internal_hidden, _ = _run_sub_blocks(internal_hidden,
                                                             self.scc_layer[
                                                             i * base_interval:i * base_interval + base_interval],
                                                             left_idx)

                    pattern_idx //= 2
                return internal_hidden, early_exits_logits

            large_hidden, large_early_logits = _run(pattern_idx=2 ** self.num_parts - 1)
            base_hidden, base_early_logits = _run(pattern_idx=0)
            random_hidden, random_early_logits = _run(
                pattern_idx=random.choice([i for i in range(0, 2 ** self.num_parts)]))
            outputs = (large_hidden,)
            all_early_logits = large_early_logits + base_early_logits + random_early_logits
            outputs = outputs + (all_early_logits,)
            return outputs

        elif self.train_agent and not self.only_large_and_exit:
            assert self.agent.action_classifier.out_features == 3, \
                "action space number is supposed to be 3: base, large and exit, current is %d" % self.agent.action_classifier.out_features

            bsz = hidden_states.size()[0]
            device = hidden_states.device
            left_idx = torch.arange(bsz, device=device)
            large_interval = self.prd_n_layer // self.num_parts
            base_interval = self.scc_n_layer // self.num_parts
            # training with a switch agent
            action_probs = ()
            actions = ()
            internal_classifier_logits = ()
            # early_exit_pairs = []
            early_exit_logits = ()
            early_exit_idxs = ()
            rnn_hidden = self.rnn_agent.reset_hidden(bsz, device,
                                                     hidden_states.dtype)  # Tuple( (1,bsz, rnn_hidden), (1, bsz, rnn_hidden))

            for i in range(self.num_parts):
                if len(hidden_states) == 0:
                    break
                # curriculum learning, from first block to last
                if i > self.cl_idx:  # and self.training:
                    action = torch.zeros((len(hidden_states),), dtype=torch.long, device=device) * 2
                    action_prob = torch.zeros((len(hidden_states), self.agent.action_classifier.out_features),
                                              device=device)
                else:
                    action_prob, rnn_hidden = self.rnn_agent(hidden_states, rnn_hidden)
                    if self.bound_alpha > 0:
                        action_prob = self.adjust_prob(action_prob)  # adjust prob distribution according to alpha
                        # policy gradient
                    if self.training:
                        m = Categorical(action_prob)
                        action = m.sample()
                    else:  # during evaluation, we do not sample but using the argmax for path selection
                        action = torch.argmax(action_prob, dim=-1)

                padded_prob = torch.ones((bsz, self.agent.action_classifier.out_features), device=device)
                padded_prob[left_idx] = action_prob
                action_probs = action_probs + (padded_prob,)

                padded_action = torch.zeros(size=(bsz,), device=device, dtype=torch.long)
                padded_action[left_idx] = action
                actions = actions + (padded_action,)

                # branchy logic
                exit_idx = left_idx[action == 0]  # using 0 for current code
                if len(exit_idx) > 0:
                    exited_logit = self.early_classifiers[i](hidden_states)[action == 0]
                    early_exit_logits = early_exit_logits + (exited_logit,)
                    early_exit_idxs = early_exit_idxs + (exit_idx,)

                #  to implement acceleration, exited examples are not supposed to continue the forward loop
                base_idx = left_idx[action == 1]
                large_idx = left_idx[action == 2]
                base_input = hidden_states[action == 1]
                large_input = hidden_states[action == 2]

                rnn_hidden = (rnn_hidden[0][:, action != 0], rnn_hidden[1][:, action != 0])

                base_hiddens = base_input
                large_hiddens = large_input

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

            outputs = outputs + (left_idx,
                                 action_probs,
                                 actions,
                                 early_exit_logits,
                                 early_exit_idxs,
                                 internal_classifier_logits
                                 )  # action_probs for computing loss
            if self.output_hidden_states:
                outputs = outputs + (all_hidden_states,)
            if self.output_attentions:
                outputs = outputs + (all_attentions,)
            return outputs  # last-layer hidden state, action_probs, (all hidden states), (all attentions)

        elif self.train_agent and self.only_large_and_exit:  # only large and exit
            assert self.simple_agent.action_classifier.out_features == 2, "only large and exit, aciton is 2"
            bsz = hidden_states.size()[0]
            device = hidden_states.device
            left_idx = torch.arange(bsz, device=device)
            large_interval = self.prd_n_layer // self.num_parts
            # training with a switch agent
            action_probs = ()
            actions = ()
            early_exit_logits = ()
            early_exit_idxs = ()

            for i in range(self.num_parts):
                action_prob = self.simple_agent(hidden_states)
                padded_prob = torch.ones((bsz, self.simple_agent.action_classifier.out_features), device=device)
                padded_prob[left_idx] = action_prob
                action_probs = action_probs + (padded_prob,)
                # policy gradient
                if self.training:
                    m = Categorical(action_prob)
                    action = m.sample()
                else:  # during evaluation, we do not sample but using the argmax for path selection
                    action = torch.argmax(action_prob, dim=-1)
                padded_action = torch.zeros(size=(bsz,), device=device, dtype=torch.long)
                padded_action[left_idx] = action
                actions = actions + (padded_action,)

                exit_idx = left_idx[action == 0]  # using 0 for current code
                if len(exit_idx) > 0:
                    exited_logit = self.early_classifiers[i](hidden_states[action == 0])
                    early_exit_logits = early_exit_logits + (exited_logit,)
                    early_exit_idxs = early_exit_idxs + (exit_idx,)

                left_idx = left_idx[action == 1]  # large action idx = 1
                # to implement acceleration, exited examples are not supposed to continue the forward loop
                hidden_states = hidden_states[action == 1]

                if len(left_idx) > 0:
                    hidden_states, blocks_outputs = _run_sub_blocks(hidden_states,
                                                                    self.layer[
                                                                    i * large_interval:i * large_interval + large_interval],
                                                                    left_idx)

                    if self.output_hidden_states:
                        all_hidden_states = all_hidden_states + (hidden_states,)  # emit for the first hidden states?
                else:
                    break  # no examples left
                # if self.output_attentions:
                #     all_attentions = all_attentions + (blocks_outputs,)

            outputs = (hidden_states,)

            outputs = outputs + (left_idx,
                                 action_probs,
                                 actions,
                                 early_exit_logits,
                                 early_exit_idxs
                                 )  # action_probs for computing loss
            if self.output_hidden_states:
                outputs = outputs + (all_hidden_states,)
            if self.output_attentions:
                outputs = outputs + (all_attentions,)
            return outputs  # last-layer hidden state, action_probs, (all hidden states), (all attentions)
        else:  # inference with compressed model
            print('else branch, usually used for normal theseus-bert inference')
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

    def critic_forward(self, hidden_states, attention_mask=None, head_mask=None,
                       encoder_hidden_states=None,
                       encoder_attention_mask=None):
        bsz = hidden_states.size()[0]
        device = hidden_states.device
        left_idx = torch.arange(bsz, device=device)
        large_interval = self.prd_n_layer // self.num_parts
        base_interval = self.scc_n_layer // self.num_parts
        early_exit_logits = ()
        early_exit_idxs = ()
        actions = ()

        rnn_hidden = self.rnn_agent.reset_hidden(bsz, device, dtype=hidden_states.dtype, requires_grad=False)

        def _run_sub_blocks(layer_hidden_states, layers, idx):
            for i, layer in enumerate(layers):
                layer_output = layer(layer_hidden_states, attention_mask[idx], None, encoder_hidden_states,
                                     encoder_attention_mask[idx] if encoder_attention_mask is not None else None)
                layer_hidden_states = layer_output[0]
            return layer_hidden_states, layer_output

        # critic_actions: bsz, num_parts( num_parts can be small than config since examples can exit early)
        for i in range(self.num_parts):
            if len(hidden_states) == 0:
                break
            # curriculum learning
            if i > self.cl_idx:  # and self.training:
                action = torch.zeros((len(hidden_states),), device=device, dtype=torch.long) * 2
            else:
                action_prob, rnn_hidden = self.rnn_agent(hidden_states, rnn_hidden)
                action = torch.argmax(action_prob, dim=-1)

            padded_action = torch.zeros((bsz,), device=device, dtype=torch.long)
            padded_action[left_idx] = action
            actions = actions + (padded_action,)
            # action: bsz,
            exit_idx = left_idx[action == 0]  # using 0 for current code
            if len(exit_idx) > 0:
                exited_logit = self.early_classifiers[i](hidden_states[action == 0])
                early_exit_logits = early_exit_logits + (exited_logit,)
                early_exit_idxs = early_exit_idxs + (exit_idx,)

            #  to implement acceleration, exited examples are not supposed to continue the forward loop
            base_idx = left_idx[action == 1]
            large_idx = left_idx[action == 2]
            base_input = hidden_states[action == 1]
            large_input = hidden_states[action == 2]

            rnn_hidden = (rnn_hidden[0][:, action != 0], rnn_hidden[1][:, action != 0])
            base_hiddens = base_input
            large_hiddens = large_input

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
        outputs = (hidden_states,)

        # critic forward return results
        outputs = outputs + (left_idx,
                             early_exit_logits,
                             early_exit_idxs,
                             actions
                             )
        return outputs

    def adjust_prob(self, prob):
        #  adjust prob to avoid extreme exploitation
        adjusted_prob = prob * self.bound_alpha + (1 - self.bound_alpha) * (1 - prob)
        return adjusted_prob


class LinearPenaltyRatioScheduler:
    def __init__(self, model, initial_penalty_ratio, linear_k, max_penalty_ratio):
        self.model = model
        self.linear_k = linear_k
        self.initial_pr = initial_penalty_ratio
        self.step_counter = 0
        self.max_penalty_ratio = max_penalty_ratio
        self.current_pr = initial_penalty_ratio

    def step(self):
        self.step_counter += 1
        current_pr = min(self.max_penalty_ratio,
                         self.linear_k * self.step_counter + self.initial_pr)
        self.model.set_path_penalty(current_pr)
        self.current_pr = current_pr
        return current_pr


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


class CurriculumLearningScheduler:
    """Scheduler for curriculum learning"""

    def __init__(self, bert_encoder: BertEncoder, initial_cl_idx=5, epoch_interval=5, num_parts=6):
        self.bert_encoder = bert_encoder
        self.epoch_counter = 0
        self.bert_encoder.set_cl_idx(initial_cl_idx)
        self.epoch_interval = epoch_interval
        self.cl_idx = initial_cl_idx
        self.total_parts = num_parts

    def step(self):
        # after epoch, decrease the cl_idx let agent learn more blocks
        self.epoch_counter += 1
        if self.epoch_counter % self.epoch_interval == 0:
            self.cl_idx += 1
            self.bert_encoder.set_cl_idx(self.cl_idx)


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
                head_mask=None, inputs_embeds=None, encoder_hidden_states=None, encoder_attention_mask=None,
                critic_forward=False):
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
        if not critic_forward:
            encoder_outputs = self.encoder(embedding_output,
                                           attention_mask=extended_attention_mask,
                                           head_mask=head_mask,
                                           encoder_hidden_states=encoder_hidden_states,
                                           encoder_attention_mask=encoder_extended_attention_mask)
        else:
            with torch.no_grad():
                encoder_outputs = self.encoder.critic_forward(embedding_output,
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
        self.use_baseline = False
        self.error_penalty = 0.0
        self.entropy_beta = 0.0
        self.global_step = 0

    def set_switch_pattern(self, switch_pattern):
        self.bert.encoder.switch_pattern = switch_pattern

    def set_path_penalty(self, penalty_ratio):
        self.path_penalty_ratio = penalty_ratio

    def set_error_penalty(self, error_penalty):
        self.error_penalty = error_penalty

    def set_entropy_beta(self, entropy_beta):
        self.entropy_beta = entropy_beta

    def set_baseline(self):
        self.use_baseline = True

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
        early_exit_idx: Tuple = None
        early_exit_logit: Tuple = None
        internal_classifier_logits = None
        critic_actions = ()
        padded_critic_actions = ()
        critic_logits = None
        critic_early_exit_idx = None
        critic_early_exit_logit = None
        if self.bert.encoder.train_agent:
            left_idx = outputs[2]
            action_probs = outputs[3]
            actions = outputs[4]
            early_exit_logit = outputs[5]
            early_exit_idx = outputs[6]
            # internal_classifier_logits = outputs[7] # we separate internal classifier trianing
        elif self.bert.encoder.train_early_exit:
            internal_classifier_logits = outputs[-1]

        if self.training:  # self-critic baseline
            critic_outputs = self.bert(input_ids,
                                       critic_forward=True,
                                       attention_mask=attention_mask,
                                       token_type_ids=token_type_ids,
                                       position_ids=position_ids,
                                       head_mask=head_mask,
                                       inputs_embeds=inputs_embeds)
            critic_pooled_output = critic_outputs[1]
            critic_pooled = self.dropout(critic_pooled_output)
            critic_logits = self.classifier(critic_pooled)

            # resemble the logits
            critic_left_idx = critic_outputs[2]
            critic_early_exit_logit = critic_outputs[3]
            critic_early_exit_idx = critic_outputs[4]
            if critic_early_exit_idx is not None and len(critic_early_exit_idx) > 0:  # if early exit happens,
                # re-order it back
                critic_total_idx = torch.cat(critic_early_exit_idx + (critic_left_idx,), dim=0)
                _, critic_order = torch.sort(critic_total_idx)
                critic_logits = torch.cat(critic_early_exit_logit + (critic_logits,), dim=0)[critic_order]
            padded_critic_actions = critic_outputs[5]
            self.global_step += 1

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        # early_exit_logit = torch.cat([p[0] for p in early_exit_pairs], dim=0)
        # early_exit_idx = torch.cat([p[1] for p in early_exit_pairs], dim=0)
        if early_exit_idx is not None and len(early_exit_idx) > 0:  # if early exit happens, re-order it back
            total_idx = torch.cat(early_exit_idx + (left_idx,), dim=0)
            _, order = torch.sort(total_idx)
            logits = torch.cat(early_exit_logit + (logits,), dim=0)[order]

        if self.bert.encoder.early_exit_idx != -1 and self.bert.encoder.train_early_exit:  # test for specific early exit
            logits = internal_classifier_logits[self.bert.encoder.early_exit_idx]
        elif self.bert.encoder.early_exit_idx == -1 and self.bert.encoder.train_early_exit:
            logits = random.choice(internal_classifier_logits)  # random choose a logit

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        paths = []
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            # internal classifier loss
            if self.bert.encoder.train_early_exit:
                early_losses = []
                for early_logits in internal_classifier_logits:  #
                    if self.num_labels == 1:
                        #  We are doing regression
                        loss_fct = MSELoss()
                        early_loss = loss_fct(early_logits.view(-1), labels.view(-1))
                    else:
                        loss_fct = CrossEntropyLoss()
                        early_loss = loss_fct(early_logits.view(-1, self.num_labels), labels.view(-1))
                    early_losses.append(early_loss)
                if not self.training and self.bert.encoder.early_exit_idx != -1:
                    loss = early_losses[self.bert.encoder.early_exit_idx]
                else:
                    loss = sum(early_losses)

            if action_probs is not None:
                bsz = logits.size()[0]
                final_decision_prob = torch.ones((bsz,), device=input_ids.device)
                path_penalty = torch.zeros((bsz,), device=input_ids.device)
                for i, (path_prob, action) in enumerate(zip(action_probs, actions)):
                    selected_path = action.unsqueeze(1)  # bsz, 1
                    if i > self.bert.encoder.cl_idx:  # and self.training:
                        prob = torch.ones_like(final_decision_prob)  # directly set the path probability to 1
                    else:
                        prob = torch.gather(path_prob, dim=-1, index=selected_path).squeeze()  # bsz
                    final_decision_prob *= prob
                    path_penalty += action  #
                    paths.append(selected_path)

                paths = torch.cat(paths, dim=-1)
                if self.global_step % 400 == 0:
                    print(paths[:20])
                    print(final_decision_prob[:20])
                # if not self.training:
                # we can add an expected saving computation here
                # print("Layer ratio: %.3f%%" % (
                #         (torch.sum(paths) / torch.sum(all_large, dtype=torch.float)).item() * 100))
                # print(paths[:4])  # sample for some path

                if self.num_labels != 1:
                    entropy_reward_fct = CrossEntropyLoss(reduction='none')
                    performance_reward = - entropy_reward_fct(logits.view(-1, self.num_labels), labels.view(-1))
                    reward = self.get_reward(logits, labels, paths, self.error_penalty)
                    if self.global_step % 400 == 0:
                        print('reward\n', reward[:20])
                    if critic_logits is not None:
                        padded_critic_paths = torch.cat(
                            [critic_action.unsqueeze(1) for critic_action in padded_critic_actions], dim=-1)
                        if self.global_step % 400 == 0:
                            print('critic path\n', padded_critic_paths[:20])
                        baseline = self.get_reward(critic_logits, labels, padded_critic_paths, self.error_penalty)
                        if self.global_step % 400 == 0:
                            print('baseline\n', baseline[:20])
                        reward = reward - baseline  # minus self-critic baseline
                else:
                    raise ValueError("Current the regression is not supported")
                    # mse_reward_fct = MSELoss(reduction='none')
                    # performance_reward = - mse_reward_fct(logits.view(-1), labels.view(-1))
                # if self.use_baseline:
                #     path_penalty = path_penalty - torch.mean(path_penalty)
                #
                penalty_reward = - path_penalty
                # reward = performance_reward + penalty_reward
                # if self.use_baseline:
                #    reward = reward - torch.mean(reward)  # minus baseline
                # final_decision_prob = final_decision_prob.clamp(1e-9, 1 - 1e-9)
                loss = - reward * torch.log(final_decision_prob)  # sum over bsz

                # entropy loss
                entropy_loss = - final_decision_prob * torch.log(final_decision_prob)
                loss = loss - self.entropy_beta * entropy_loss  # minus action entropy

                loss = loss.mean()
                # if self.training:
                #     loss = loss + internal_loss - reward
                # else:
                # decouple early exit training & agent training
                # loss = - reward

                outputs = outputs + (
                    final_decision_prob, torch.mean(penalty_reward),
                    torch.mean(performance_reward), paths,)
                # minus reward + penalty
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)

    def get_reward(self, logits, labels, paths, error_penalty=-0.1):
        assert self.num_labels != 1
        predicted_labels = torch.argmax(logits, dim=-1)  # bsz
        correct_labels = predicted_labels.eq(labels)  #
        # paths = torch.cat(paths, dim=-1)  # bsz, num_parts
        expected_saving = paths.sum(dim=1).type_as(logits) / (self.bert.encoder.num_parts * 2)
        sparse_reward = 1 - expected_saving  # torch.pow(expected_saving, 2)
        error_penalty = torch.ones_like(sparse_reward) * error_penalty
        reward = torch.where(correct_labels, sparse_reward, error_penalty)
        return reward


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
