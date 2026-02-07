import torch
import json
from msa_transformer import MSABertForPretrain, MSABertConfig
import torch.nn as nn
import fm

# Paths (modify as needed)
config_path = "../MSA_BERT/config.json"
bert_ckpt_dir = "../MSA_BERT"
linear_ckpt_path = "../pretrained/latest_linear_uu_1226.pth"
resnet_ckpt_path = "../pretrained/res_model_uu_1226.pth"
out_path = "../pretrained/ss_bundle.pt"

def build_resnet():
    model, _ = fm.downstream.build_rnafm_resnet(type="ss")
    resnet = model.downstream_modules['pc-resnet_1_sym_first:r-ss']
    resnet.input_msa = True
    resnet.msa_embed_dim_in = resnet.embed_dim
    resnet.msa_embed_dim_out = resnet.embed_dim // resnet.attention_heads
    resnet.msa_q_proj = nn.Linear(resnet.msa_embed_dim_in, resnet.msa_embed_dim_out)
    resnet.msa_k_proj = nn.Linear(resnet.msa_embed_dim_in, resnet.msa_embed_dim_out)
    resnet.depth_reduction = 'mean'
    return resnet

# 1. Load config.json as dict
with open(config_path, "r") as f:
    config_dict = json.load(f)

# 2. Load BERT weights
bert_model = MSABertForPretrain.from_pretrained(bert_ckpt_dir)
bert_state_dict = bert_model.state_dict()

# 3. Load linear weights
linear = nn.Linear(768, 640)
linear.load_state_dict(torch.load(linear_ckpt_path, map_location="cpu"))
linear_state_dict = linear.state_dict()

# 4. Load resnet weights
resnet = build_resnet()
resnet.load_state_dict(torch.load(resnet_ckpt_path, map_location="cpu"))
resnet_state_dict = resnet.state_dict()

# 5. Save all to one bundle
bundle = {
    "config": config_dict,
    "bert_state_dict": bert_state_dict,
    "linear_state_dict": linear_state_dict,
    "resnet_state_dict": resnet_state_dict
}
torch.save(bundle, out_path)
print(f"Bundle saved to {out_path}")
