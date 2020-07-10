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

class MixedBertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, model_base, model_large, switch_rate=0.5):
        super().__init__(model_base.config)
        self.bernoulli = Bernoulli(torch.tensor([switch_rate]))
        self.model_base = model_base
        self.model_large = model_large

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


def switchable_forward(model_base, model_large, inputs_embeds, bernouali):
    # do embedding outside
    pass
