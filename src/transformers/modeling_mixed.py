from torch.distributions import Bernoulli

from src.transformers.modeling_bert import *

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
    def __init__(self, model_base, model_large, switch_rate=0.5):
        super().__init__()
        self.bernoulli = Bernoulli(torch.tensor([switch_rate]))
        self.model_base = model_base
        self.model_large = model_large
        self.config = BertConfig()

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
        if self.bernoulli.sample() == 1:  # switch base or large bert model
            outputs = self.model_base(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                mlp_mask=mlp_mask,
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
                mlp_mask=mlp_mask,
                labels=labels
            )

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


def switchable_forward(model_base, model_large, inputs_embeds, bernouali):
    # do embedding outside
    pass
