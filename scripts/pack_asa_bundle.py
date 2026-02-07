import torch
import json
from msa_transformer import MSABertForPretrain, MSABertConfig
from predict_asa import CNNTokenClassifier

# Paths (modify as needed)
config_path = "../MSA_BERT/config.json"
bert_ckpt_dir = "../MSA_BERT"
cnn_ckpt_path = "../pretrained/RMSA_ASA.pt"
out_path = "../pretrained/asa_bundle.pt"

# 1. Load config.json as dict
with open(config_path, "r") as f:
    config_dict = json.load(f)

# 2. Load BERT weights
bert_model = MSABertForPretrain.from_pretrained(bert_ckpt_dir)
bert_state_dict = bert_model.state_dict()

# 3. Load CNN weights
cnn_model = CNNTokenClassifier(input_dim=768, hidden_dim=256)
cnn_state_dict = torch.load(cnn_ckpt_path, map_location="cpu")

# 4. Save all to one bundle
bundle = {
    "config": config_dict,
    "bert_state_dict": bert_state_dict,
    "cnn_state_dict": cnn_state_dict
}
torch.save(bundle, out_path)
print(f"Bundle saved to {out_path}")
