from transformers import PretrainedConfig, PreTrainedModel, GPT2Config,GPT2LMHeadModel
# from modeling_gpt2 import GPT2LMHeadModel
import torch.nn as nn
from esm import Alphabet,MSATransformer
from transformers.activations import ACT2FN
import torch
import torch.nn.functional as F
import json
class MSABertConfig(PretrainedConfig):

    model_type = "msa_bert"

    def __init__(
        self,
        layers=12,
        embed_dim=768,
        logit_bias=False,
        ffn_embed_dim=3072,
        attention_heads=12,
        dropout=0.1,
        attention_dropout=0.1,
        activation_dropout=0.1,
        max_tokens=2**14,
        padding_idx=0,
        max_positions=1024,
        embed_positions_msa=True,
        alphabet_config=None,
        alphabet_size=None,
        **kwargs,
    ):
        if alphabet_config is None:
            alphabet_config = {
            "standard_toks": ["A", "G","T",  "C", ".", "-"],
            "prepend_toks": ["<cls>", "<pad>", "<unk>"],
            "append_toks": ["<mask>"],
            "prepend_bos": True,
            "append_eos": False,
            "use_msa": True
        }
        self.padding_idx = padding_idx
        super().__init__( **kwargs)
        self.embed_positions_msa = embed_positions_msa
        self.alphabet_config = alphabet_config
        alphabet=Alphabet(**alphabet_config)
        self.alphabet_size = len(alphabet)
        self.layers = layers
        self.embed_dim = embed_dim
        self.ffn_embed_dim = ffn_embed_dim
        self.attention_heads = attention_heads
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.max_tokens = max_tokens
        self.logit_bias = logit_bias
        self.max_positions = max_positions

class MSABertForPretrain(PreTrainedModel):
    config_class = MSABertConfig
    def __init__(self, config: MSABertConfig):
        super().__init__(config)
        self.model = MSATransformer(
            config,Alphabet(**config.alphabet_config)
        )
        self.init_weights()

    def compute_loss(self, logits, labels):
        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)  # ignore_index is used for padding tokens

        logits_flat = logits.view(-1, logits.size(-1))  # Shape: (batch_size * seq_length, vocab_size)
        labels_flat = labels.view(-1)  # Shape: (batch_size * seq_length)


        masked_labels = labels_flat[labels_flat != -100]  # Remove padding tokens (they are labeled -100)
        masked_logits = logits_flat[labels_flat != -100]  # Filter out non-masked token logits

        loss = loss_fct(masked_logits, masked_labels)
        return loss

    def forward(self, input_ids=None,labels=None, **kwargs):
        output=self.model(input_ids,repr_layers=[self.config.layers],**kwargs)
        logits=output['logits']
        hidden_states=output['representations'][self.config.layers]
        output['hidden_states']=hidden_states
        if labels is not None:
            loss=self.compute_loss(logits,labels)
            output['loss']=loss
        return output

class MSATransformerWithGPT2Config(PretrainedConfig):
        model_type = "msa_transformer"
        def __init__(
                self,
                msa_bert_config: MSABertConfig=None,
                gpt2_config: GPT2Config=None,
                **kwargs,
        ):
            super().__init__(**kwargs)
            self.msa_bert_config=msa_bert_config
            self.gpt2_config=gpt2_config
        def from_pretrained(pretrained_model_name_or_path, **kwargs):
            config = json.load(open(pretrained_model_name_or_path))
            msa_bert_config=MSABertConfig(**config['msa_bert_config'])
            gpt2_config=GPT2Config(**config['gpt2_config'])
            return MSATransformerWithGPT2Config(msa_bert_config=msa_bert_config,gpt2_config=gpt2_config,**kwargs)


class MultiModalProjector(nn.Module):
    def __init__(self, config: MSATransformerWithGPT2Config):
        super().__init__()

        self.linear_1 = nn.Linear(config.msa_bert_config.embed_dim, config.msa_bert_config.ffn_embed_dim, bias=True)
        self.act = ACT2FN[config.gpt2_config.activation_function]
        self.linear_2 = nn.Linear(config.msa_bert_config.ffn_embed_dim, config.gpt2_config.n_embd, bias=True)

    def forward(self, image_features):
        hidden_states = self.linear_1(image_features)
        hidden_states = self.act(hidden_states)
        # pool
        hidden_states = hidden_states.mean(dim=1)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states
    
class MSATransformerWithGPT2(PreTrainedModel):
    config_class = MSATransformerWithGPT2Config
    def __init__(self, config: MSATransformerWithGPT2Config,**kwargs):
        super().__init__(config,**kwargs)
        self.msa_bert_model = MSABertForPretrain(config.msa_bert_config)
        self.gpt2_model= GPT2LMHeadModel(config.gpt2_config)
        self.projector=MultiModalProjector(config)
        self.init_weights()

    def compute_loss(self, logits, labels):
        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        # shift labels to the right
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        return loss

    def forward(self, input_ids=None,labels=None, **kwargs):
        output=self.msa_bert_model(input_ids,**kwargs)
        hidden_states=output['hidden_states']
        # shape=(batch_size, msa_len, seq_len,embed_dim)
        # len_per_seqs=[hidden_states.shape[1],]
        # hidden_states=hidden_states.reshape(hidden_states.shape[0],-1,hidden_states.shape[-1])
        hidden_states=self.projector(hidden_states)
        if labels is not None:
            to_add_label=labels[:,input_ids.shape[-1]+1:]
            bos_label=torch.tensor([self.gpt2_model.config.bos_token_id]*to_add_label.shape[0],device=self.gpt2_model.device).unsqueeze(1)
            labels_embeds = self.gpt2_model.transformer.wte(torch.cat((bos_label,to_add_label),dim=1))
            hidden_states = torch.cat((hidden_states, labels_embeds), dim=1)
        output=self.gpt2_model(inputs_embeds=hidden_states,labels=labels,**kwargs)
        logits=output['logits']

        if labels is not None:
            loss=self.compute_loss(logits,labels)
            output['loss']=loss
        return output
    
    def generate(self, input_ids=None,temperature=1.0, top_k=800, **kwargs):
        device = self.gpt2_model.device
        input_ids = torch.tensor(input_ids, dtype=torch.long, device=device)
        max_new_length = input_ids.shape[-1]-1
        with torch.no_grad():
            hidden_states = self.msa_bert_model(input_ids,**kwargs)['hidden_states']
            hidden_states=self.projector(hidden_states)
            bos_embeds = self.gpt2_model.transformer.wte(torch.tensor([self.gpt2_model.config.bos_token_id],device=device)).unsqueeze(1)
            next_token=None
            past_key_values = None  
            generated_ids = None
            for _ in range(max_new_length):
                # 前向传播，获取当前的 logits 和 past_key_values
                outputs = self.gpt2_model(
                    inputs_embeds=torch.cat((hidden_states,bos_embeds),dim=1) if past_key_values is None else None,
                    input_ids=next_token,  # 只传递最新生成的token
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,

                )
                
                next_token_logits = outputs.logits[:, -1, :] / temperature
                past_key_values = outputs.past_key_values

                # 贪婪搜索或top-k采样
                if top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k, dim=-1)
                    probabilities = F.softmax(top_k_logits, dim=-1)
                    next_token = top_k_indices.gather(-1, torch.multinomial(probabilities, 1))
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)


                # 连接生成的 token 到输入序列中
                if generated_ids is None:
                    generated_ids = next_token
                else:
                    generated_ids = torch.cat((generated_ids, next_token), dim=1)

                # 检查是否生成了结束符 <EOS>（假设 EOS token 的索引为 2）
                if torch.all(next_token == self.gpt2_model.config.eos_token_id):
                    break
            
            return generated_ids