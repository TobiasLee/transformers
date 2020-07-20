from torch.distributions import Bernoulli
import random

from src.transformers import AutoConfig
from src.transformers.modeling_bert import *
from src.transformers.modeling_utils import ModuleUtilsMixin

# {  BERT-large
#   "architectures": [
#     "BertForMaskedLM"
#   ],
#   "attention_probs_dropout_prob": 0.1,
#   "hidden_act": "gelu",
#   "hidden_dropout_prob": 0.1,
#   "hidden_size": 1024,
#   "initializer_range": 0.02,
#   "intermediate_size": 4096,
#   "layer_norm_eps": 1e-12,
#   "max_position_embeddings": 512,
#   "model_type": "bert",
#   "num_attention_heads": 16,
#   "num_hidden_layers": 24,
#   "pad_token_id": 0,
#   "type_vocab_size": 2,
#   "vocab_size": 30522
# }

# {  BERT-base
#   "architectures": [
#     "BertForMaskedLM"
#   ],
#   "attention_probs_dropout_prob": 0.1,
#   "hidden_act": "gelu",
#   "hidden_dropout_prob": 0.1,
#   "hidden_size": 768,
#   "initializer_range": 0.02,
#   "intermediate_size": 3072,
#   "layer_norm_eps": 1e-12,
#   "max_position_embeddings": 512,
#   "model_type": "bert",
#   "num_attention_heads": 12,
#   "num_hidden_layers": 12,
#   "pad_token_id": 0,
#   "type_vocab_size": 2,
#   "vocab_size": 30522
# }
WEIGHTS_NAME = "pytorch_model.bin"


class MixedBertForSequenceClassification(nn.Module):
    base_model_prefix = "bert"

    def __init__(self, model_base, model_large, switch_rate=0.5, mode='random', config=BertConfig()):
        super().__init__()
        self.bernoulli = Bernoulli(torch.tensor([switch_rate]))
        self.model_base = model_base
        self.model_large = model_large
        self.config = config
        self.mode = mode

    @classmethod
    def set_prefix(cls, prefix):
        cls.base_model_prefix = prefix

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            mlp_mask=None
    ):
        if self.mode == 'random':
            if self.bernoulli.sample() == 1:  # switch base or large bert model
                outputs = self.model_base(
                    input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    position_ids=position_ids,
                    head_mask=head_mask,
                    inputs_embeds=inputs_embeds,
                    # mlp_mask=mlp_mask,
                    labels=labels
                )
            else:
                outputs = self.model_large(
                    input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    position_ids=position_ids,
                    head_mask=head_mask,
                    inputs_embeds=inputs_embeds,
                    # mlp_mask=mlp_mask,
                    labels=labels
                )
        elif self.mode == 'large':
            outputs = self.model_large(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                # mlp_mask=mlp_mask,
                labels=labels
            )
        elif self.mode == 'base':
            outputs = self.model_base(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                # mlp_mask=mlp_mask,
                labels=labels
            )
        else:
            raise ValueError("Unsupported mix mode: selected between: random/large/base")

        return outputs  # (loss), logits, (hidden_states), (attentions)

    def save_pretrained(self, save_directory):
        """ Save a model and its configuration file to a directory, so that it
            can be re-loaded using the `:func:`~transformers.PreTrainedModel.from_pretrained`` class method.

            Arguments:
                save_directory: directory to which to save.
        """
        assert os.path.isdir(
            save_directory
        ), "Saving path should be a directory where the model and configuration can be saved"

        # Only save the model itself if we are using distributed training
        model_to_save = self.module if hasattr(self, "module") else self

        # Attach architecture to the config
        model_to_save.config.architectures = [model_to_save.__class__.__name__]

        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(save_directory, WEIGHTS_NAME)
        model_to_save.config.save_pretrained(save_directory)
        torch.save(model_to_save.state_dict(), output_model_file)

        logger.info("Model weights saved in {}".format(output_model_file))

    @classmethod
    def from_pretrained(cls, path, model_base, model_large, mode='random'):
        archive_file = os.path.join(path, WEIGHTS_NAME)

        try:
            state_dict = torch.load(archive_file, map_location="cpu")
        except Exception:
            raise OSError(
                "Unable to load weights from pytorch checkpoint file. "
                "If you tried to load a PyTorch model from a TF 2.0 checkpoint, please set from_tf=True. "
            )
        model = cls(model_base, model_large)

        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = None
            if "gamma" in key:
                new_key = key.replace("gamma", "weight")
            if "beta" in key:
                new_key = key.replace("beta", "bias")
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)

        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, "_metadata", None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        # PyTorch's `_load_from_state_dict` does not copy parameters in a module's descendants
        # so we need to apply the function recursively.
        def load(module: nn.Module, prefix=""):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs,
            )
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + ".")

        # Make sure we are able to load base models as well as derived models (with heads)
        start_prefix = ""
        model_to_load = model
        has_prefix_module = any(s.startswith(cls.base_model_prefix) for s in state_dict.keys())
        if not hasattr(model, cls.base_model_prefix) and has_prefix_module:
            start_prefix = cls.base_model_prefix + "."
        if hasattr(model, cls.base_model_prefix) and not has_prefix_module:
            model_to_load = getattr(model, cls.base_model_prefix)

        load(model_to_load, prefix=start_prefix)

        if model.__class__.__name__ != model_to_load.__class__.__name__:
            base_model_state_dict = model_to_load.state_dict().keys()
            head_model_state_dict_without_base_prefix = [
                key.split(cls.base_model_prefix + ".")[-1] for key in model.state_dict().keys()
            ]

            missing_keys.extend(head_model_state_dict_without_base_prefix - base_model_state_dict)

        if len(missing_keys) > 0:
            logger.info(
                "Weights of {} not initialized from pretrained model: {}".format(
                    model.__class__.__name__, missing_keys
                )
            )
        if len(unexpected_keys) > 0:
            logger.info(
                "Weights from pretrained model not used in {}: {}".format(
                    model.__class__.__name__, unexpected_keys
                )
            )
        if len(error_msgs) > 0:
            raise RuntimeError(
                "Error(s) in loading state_dict for {}:\n\t{}".format(
                    model.__class__.__name__, "\n\t".join(error_msgs)
                )
            )
        model.model_base.tie_weights()
        model.model_large.tie_weights()
        model.eval()
        model.mode = mode

        return model


class RandomPathModel(MixedBertForSequenceClassification):
    def __init__(self, model_base, model_large, switch_rate=0.5, num_parts=3,
                 base_model_name='bert',
                 large_model_name='bert'
                 ):
        super(RandomPathModel, self).__init__(model_base, model_large, switch_rate)
        self.num_parts = num_parts
        self.base_model_name = base_model_name
        self.large_model_name = large_model_name
        self.output_attentions = self.model_base.config.output_attentions
        self.output_hidden_states = self.model_base.config.output_hidden_states
        self.mixed_bert = MixedBert(model_base, model_large, num_parts,
                                    self.base_model_name, self.large_model_name)
        # final layer
        self.large_classifier = model_large.classifier
        self.base_classifier = model_base.classifier
        self.base_dropout = getattr(self.model_base, 'dropout', None)
        self.large_dropout = getattr(self.model_large, 'dropout', None)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            mlp_mask=None
    ):
        # we divide the large & base model into 3 parts
        outputs = self.mixed_bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            # mlp_mask=mlp_mask
        )
        last_selected_block_idx = outputs[-1][-1]  # selected idx

        # if hidden_num == large_hidden:
        if self.large_dropout is not None:  # bert model
            pooled_output = outputs[1]
            if last_selected_block_idx == 1:
                pooled_output = self.large_dropout(pooled_output)
                logits = self.large_classifier(pooled_output)
            else:
                pooled_output = self.base_dropout(pooled_output)
                logits = self.base_classifier(pooled_output)
        else:  # roberta & distill-roberta model
            sequence_output = outputs[0]
            if last_selected_block_idx == 1:
                logits = self.large_classifier(sequence_output)
            else:
                logits = self.base_classifier(sequence_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.model_base.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.model_base.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


class MixedBert(nn.Module, ModuleUtilsMixin):
    def __init__(self, model_base, model_large, num_parts,
                 base_model_name='bert',
                 large_model_name='bert'):
        super(MixedBert, self).__init__()
        self.add_module("model_base", model_base)
        self.add_module("model_large", model_large)
        # self.model_base = model_base
        # self.model_large = model_large
        base_model_handler = getattr(self.model_base, base_model_name)
        large_model_handler = getattr(self.model_large, large_model_name)
        self.base_pooler = base_model_handler.pooler
        self.large_pooler = large_model_handler.pooler
        self.base_embeddings = base_model_handler.embeddings
        self.large_embeddings = large_model_handler.embeddings
        self.mixed_encoder = MixedEncoder(self.model_base, self.model_large, num_parts,
                                          base_model_name,
                                          large_model_name)
        self.config = model_base.config

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                mlp_mask=None
                ):
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
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
        if mlp_mask is None:
            mlp_mask = [None] * self.config.num_hidden_layers
        # print(device)
        # layers = self.mixed_encoder.get_switchable_forward()
        # layers = [module.to(device) for module in layers]

        # if layers[0].attention.self.query.in_features == self.model_base.config.hidden_size:
        base_embedding_output = self.base_embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds
        )
        # else:
        large_embedding_output = self.large_embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds
        )
        encoder_outputs = self.mixed_encoder(
            base_embedding_output,
            large_embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            # mlp_mask=mlp_mask,
            layers=None,
        )
        sequence_output = encoder_outputs[0]
        hidden_num = sequence_output.size()[-1]

        # pooler need to be switch according to the hidden size
        if hidden_num == self.base_pooler.dense.in_features:
            pooled_output = self.base_pooler(sequence_output)
        else:
            pooled_output = self.large_pooler(sequence_output)
        # add hidden_states and attentions if they are here
        outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]
        return outputs


class MixedEncoder(nn.Module):
    def __init__(self, model_base, model_large, num_parts=3,
                 base_model_name='bert',
                 large_model_name='bert'):
        super(MixedEncoder, self).__init__()
        self.model_base = model_base
        self.model_large = model_large
        self.num_parts = num_parts
        # extra transformation layers
        self.lo2hi_layers = nn.ModuleList([nn.Linear(self.model_base.config.hidden_size,
                                                     self.model_large.config.hidden_size)
                                           for _ in range(num_parts)])
        self.hi2lo_layers = nn.ModuleList([nn.Linear(self.model_large.config.hidden_size,
                                                     self.model_base.config.hidden_size)
                                           for _ in range(num_parts)])
        # divide into parts
        self.base_interval = self.model_base.config.num_hidden_layers // num_parts
        self.large_interval = self.model_large.config.num_hidden_layers // num_parts
        # a ref to the underlying model: e.g. bert/roberta/distilroberata
        base_model_handler = getattr(self.model_base, base_model_name)
        large_model_handler = getattr(self.model_large, large_model_name)

        self.base_parts = nn.ModuleList([base_model_handler.encoder.layer[i:i + self.base_interval]
                                         for i in
                                         range(0, self.model_base.config.num_hidden_layers, self.base_interval)])

        self.large_parts = nn.ModuleList([large_model_handler.encoder.layer[i:i + self.large_interval]
                                          for i in
                                          range(0, self.model_large.config.num_hidden_layers, self.large_interval)])
        # configs
        self.output_attentions = base_model_handler.config.output_attentions
        self.output_hidden_states = large_model_handler.config.output_hidden_states

    def forward(self,
                base_embeddings,
                large_embeddings,
                attention_mask=None,
                head_mask=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                mlp_mask=None,
                layers=None,
                ):

        # dynamic_encoder_layers = self.get_switchable_forward()
        all_hidden_states = ()
        all_attentions = ()
        # print('layer number for this time: %d' % len(layers))
        layers = []
        selected_path = []
        # print(self.base_parts)
        # print(len(self.base_parts))
        # move the random path logit into forward to avoid multi-gpu bug
        for i in range(self.num_parts):
            # select between large or base blocks
            # TODO:  replace random choice with instance-level metric
            selected_idx = random.choice([0, 1])  # 0:base, 1:large

            # add feature transformation between mismatch blocks
            if len(selected_path) != 0 and selected_idx != selected_path[-1]:
                # select a large block if previous is small, add a lo2hi transformation
                if selected_idx > selected_path[-1]:
                    layers.append(self.lo2hi_layers[i])
                else:
                    layers.append(self.hi2lo_layers[i])

            selected_path.append(selected_idx)
            layers.extend(self.base_parts[i] if selected_idx == 0
                          else self.large_parts[i])

        selected_path = torch.tensor(selected_path, device=base_embeddings.device)
        if selected_path[0] == 0:
            hidden_states = base_embeddings
        else:
            hidden_states = large_embeddings
        for i, layer_module in enumerate(layers):
            # print(i, hidden_states.device, next(layer_module.parameters()).device)
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            if isinstance(layer_module, nn.Linear):
                layer_outputs = (layer_module(hidden_states), None)
            else:
                # we have to set head_mask & mlp_mask to None
                layer_outputs = layer_module(
                    hidden_states, attention_mask, None, encoder_hidden_states, encoder_attention_mask  # , None
                )
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
        # last-layer hidden state, (all hidden states), (all attentions), (selected_path)
        return outputs + (selected_path,)

    def get_switchable_forward(self):  # get a switched encoder layer, cannot work under multi-gpu setting
        # forward = nn.ModuleList()
        forward = []
        # print(self.base_parts)
        # print(len(self.base_parts))
        for i in range(self.num_parts):
            # select between large or base blocks
            # TODO:  replace random choice with instance-level metric
            selected = random.choice([self.base_parts[i], self.large_parts[i]])
            pre_hidden = forward[-1].output.dense.out_features if len(forward) != 0 else None
            next_hidden = selected[-1].output.dense.out_features
            # add feature transformation between mismatch blocks
            if pre_hidden is not None and next_hidden != pre_hidden:
                if next_hidden > pre_hidden:
                    forward.append(self.lo2hi_layers[i])
                else:
                    forward.append(self.hi2lo_layers[i])
            forward.extend(selected)
        return forward  # return a mixed forward encoder


class Classifier(nn.Module):
    def __init__(self, config):
        super(Classifier, self).__init__()
        self.pooler = BertPooler(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, hidden_states):
        # Pooler
        pooler_output = self.pooler(hidden_states)  # can use multiple-pooler ?
        pooled_output = self.dropout(pooler_output)
        logits = self.classifier(pooled_output)

        return logits

class BranchyBert(MixedBert):
    def __init__(self, model_base, model_large, num_parts,
                 base_model_name='bert',
                 large_model_name='bert',
                 entropy_lo_threshold=0.5,
                 entropy_hi_threshold=1.0):
        super(BranchyBert, self).__init__(model_base, model_large, num_parts,
                                          base_model_name,
                                          large_model_name)
        self.num_parts = num_parts
        self.base_early_classifiers = nn.ModuleList([
            Classifier(model_base.config) for _ in range(num_parts)
        ])
        self.large_early_classifiers = nn.ModuleList([
            Classifier(model_large.config) for _ in range(num_parts)
        ])
        self.entropy_lo_threshold = entropy_lo_threshold
        self.entropy_hi_threshold = entropy_hi_threshold
        self.output_attentions = self.model_base.config.output_attentions
        self.output_hidden_states = self.model_base.config.output_hidden_states
    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                mlp_mask=None
                ):
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

        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
        if mlp_mask is None:
            mlp_mask = [None] * self.config.num_hidden_layers

        base_embedding_output = self.base_embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds
        )
        # else:
        large_embedding_output = self.large_embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds
        )

        # dynamic logic
        all_hidden_states = ()
        all_attentions = ()

        def _entropy(x):
            # TODO: check for shape match
            # x: torch.Tensor, logits BEFORE softmax
            exp_x = torch.exp(x)
            A = torch.sum(exp_x, dim=1)  # sum of exp(x_i)
            B = torch.sum(x * exp_x, dim=1)  # sum of x_i * exp(x_i)
            return torch.log(A) - B / A

        def _run_sub_blocks(layer_hidden_states, layers, idx):
            for i, layer in enumerate(layers):
                # print('layer :%d' %i )
                layer_output = layer(layer_hidden_states, extended_attention_mask[idx], None, encoder_hidden_states,
                                     encoder_extended_attention_mask)
                layer_hidden_states = layer_output[0]
            return layer_hidden_states, layer_output

        # how to decide with embedding to use ?
        # instance-wise
        selected_path = []
        bsz = input_shape[0]
        base_hidden_states = base_embedding_output
        large_hidden_states = large_embedding_output
        idx = torch.arange(bsz)
        base_idx = torch.arange(bsz)
        large_idx = torch.arange(bsz)
        logits = []
        for i in range(self.num_parts):
            # print('num parts :%d ' % i)
            # select between large & small path according to classifier entropy
            if i == 0:
                # randomly selected the first layer
                path = torch.randint(2, size=(bsz,))
                base_input = base_embedding_output[path == 0]
                large_input = large_embedding_output[path == 1]
                base_idx = idx[path == 0]
                large_idx = idx[path == 1]
                base_hidden_states, base_layer_outputs = _run_sub_blocks(base_input,
                                                                         self.mixed_encoder.base_parts[i],
                                                                         base_idx)
                large_hidden_states, large_layer_outputs = _run_sub_blocks(large_input,
                                                                           self.mixed_encoder.large_parts[i],
                                                                           large_idx)

                selected_path.append(path)

            else:
                base_logits = self.base_early_classifiers[i](base_hidden_states)
                large_logits = self.large_early_classifiers[i](large_hidden_states)

                total_logits = torch.cat((base_logits, large_logits), dim=0)
                _, order = torch.sort(torch.cat((base_idx, large_idx), dim=0))
                ordered_logits = total_logits[order]
                logits.append(ordered_logits)  # save classifier for training

                # Switch Logic
                # compute entropy
                base_ent = _entropy(base_logits)
                large_ent = _entropy(large_logits)
                #print("base entropy size:", base_ent.size()[0])
                # print("large entropy size:", large_ent.size()[0])
                # bool indicator
                still_base_entry = base_ent < self.entropy_lo_threshold
                still_large_entry = large_ent >= self.entropy_lo_threshold
                base2large_entry = base_ent >= self.entropy_lo_threshold
                large2base_entry = large_ent < self.entropy_lo_threshold

                still_base_idx = base_idx[still_base_entry]
                still_large_idx = large_idx[still_large_entry]

                base2large_idx = base_idx[base2large_entry]
                large2base_idx = large_idx[large2base_entry]

                large2base_hiddens = self.mixed_encoder.hi2lo_layers[i](
                    large_hidden_states[large2base_entry]
                )
                base2large_hiddens = self.mixed_encoder.lo2hi_layers[i](
                    base_hidden_states[base2large_entry]
                )

                base_hidden_states = torch.cat((
                    base_hidden_states[still_base_entry], large2base_hiddens
                ), dim=0)
                base_idx = torch.cat((still_base_idx, large2base_idx), dim=0)

                large_hidden_states = torch.cat((
                    large_hidden_states[still_large_entry], base2large_hiddens
                ), dim=0)
                large_idx = torch.cat((still_large_idx, base2large_idx), dim=0)
                
                # print('bp1')
                base_hidden_states, base_layer_outputs = _run_sub_blocks(base_hidden_states,
                                                                         self.mixed_encoder.base_parts[i],
                                                                         base_idx)
                #print('bp2')
                large_hidden_states, large_layer_outputs = _run_sub_blocks(large_hidden_states,
                                                                           self.mixed_encoder.large_parts[i],
                                                                           large_idx)
                # print('bp3')
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (
                    (base_hidden_states, large_hidden_states),)  # emit for the first hidden states?
            if self.output_attentions:
                all_attentions = all_attentions + ((base_layer_outputs[1], large_layer_outputs[1]),)

        outputs = ((base_hidden_states, base_idx, large_hidden_states, large_idx),)

        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        outputs = outputs + (logits, )
        return outputs  # last-layer hidden state, (all hidden states), (all attentions), (IC_logits), (selected_path)


class BranchyModel(MixedBertForSequenceClassification):
    def __init__(self, model_base, model_large, switch_rate=0.5, num_parts=3,
                 base_model_name='bert',
                 large_model_name='bert',
                 entropy_threshold=0.5
                 ):
        super(BranchyModel, self).__init__(model_base, model_large)
        self.base_model_name = base_model_name
        self.large_model_name = large_model_name
        self.output_attentions = self.model_base.config.output_attentions
        self.output_hidden_states = self.model_base.config.output_hidden_states
        self.branchy_bert = BranchyBert(model_base, model_large, num_parts,
                                        self.base_model_name, self.large_model_name,
                                        entropy_lo_threshold=entropy_threshold)
        # final layer
        self.large_classifier = model_large.classifier
        self.base_classifier = model_base.classifier
        self.base_dropout = getattr(self.model_base, 'dropout', None)
        self.large_dropout = getattr(self.model_large, 'dropout', None)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            mlp_mask=None
    ):
        outputs = self.branchy_bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        base_hidden_states, base_idx, large_hidden_states, large_idx = outputs[0]
        internal_classifier_logits = outputs[-1]
        if self.large_dropout is not None:  # bert model, pooling logic
            base_pooled = self.branchy_bert.base_pooler(base_hidden_states)
            large_pooled = self.branchy_bert.large_pooler(large_hidden_states)
            base_pooled = self.base_dropout(base_pooled)
            large_pooled = self.large_dropout(large_pooled)
            large_logits = self.large_classifier(base_pooled)
            base_logits = self.base_classifier(large_pooled)

        else:  # roberta & distill-roberta model
            large_logits = self.large_classifier(large_hidden_states)
            base_logits = self.base_classifier(base_hidden_states)

        # concat
        logits = torch.cat((base_logits, large_logits), dim=0)
        idx = torch.cat((base_idx, large_idx), dim=0)
        _, order = torch.sort(idx)
        logits = logits[order]  # order it back
        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.model_base.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))

            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.model_base.num_labels), labels.view(-1))

            # this kd can be unsupervised
            kd_loss = MSELoss()
            for internal_logit in internal_classifier_logits:
                loss += kd_loss(internal_logit.view(-1), logits.view(-1))  # teacher MSE loss for knowledge kd
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


if __name__ == '__main__':
    lo2hi_layers = nn.ModuleList([nn.Linear(123, 345) for _ in range(3)])
    hi2lo_layers = nn.ModuleList([nn.Linear(345, 123) for _ in range(3)])
    config_base = AutoConfig.from_pretrained(
        'bert-base-cased',
        num_labels=2,
        finetuning_task="mrpc",
        cache_dir=None,
    )
    config_large = AutoConfig.from_pretrained(
        'bert-large-cased',
        num_labels=2,
        finetuning_task="mrpc",
        cache_dir=None,
    )
    # model_base = BertForSequenceClassification(config_base)
    # model_large = BertForSequenceClassification(config_large)
    # switchable_forward(model_base, model_large, lo2hi_layers, hi2lo_layers)
