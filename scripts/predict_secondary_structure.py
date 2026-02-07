import os
import argparse
import random
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import fm
from msa_transformer import MSABertConfig, MSABertForPretrain
import json

def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)  # CPU-only run
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.deterministic = True  # CUDA-only
    # torch.backends.cudnn.benchmark = False     # CUDA-only

def load_a3m_file(a3m_path):
    names = []
    sequences = []

    with open(a3m_path, "r") as f:
        lines = f.readlines()[:64]  # take only the first 256 lines

    for i in range(0, len(lines), 2):  # step by 2: one header, one sequence
        name_line = lines[i].strip()
        seq_line = lines[i+1].strip() if i+1 < len(lines) else ""
        if name_line.startswith(">"):
            names.append(name_line[1:])  # remove '>'
            sequences.append(seq_line)
    return names,sequences
class RNADataLoader(Dataset):
    def __init__(self, a3m_dir):
        self.a3m_dir = a3m_dir
        self._filestems = sorted(
            os.path.splitext(f)[0] for f in os.listdir(self.a3m_dir) if f.endswith(".a3m")
        )
        # Initialize model alphabet for batch conversion
        _, self.alphabet = fm.downstream.build_rnafm_resnet(type="ss")
        self.batch_converter = self.alphabet.get_batch_converter()

    def __len__(self):
        return len(self._filestems)
    
    def __getitem__(self, idx):

        filename = self._filestems[idx]
        a3m_path = os.path.join(self.a3m_dir, f"{filename}.a3m")
        names,sequences=load_a3m_file(a3m_path)
        return {
            "seq_name": names,
            "sequence": sequences,
            "filename": filename  # Add filename to each sample
        }
        
def custom_collate_fn(batch, batch_converter):
    """
        Custom collate_fn for a3m sequences only:
            - Use batch_converter to build batch_tokens for all sequences
    """
    # 1. Build [(name, sequence), ...] and run batch_converter
    names,seqs=batch[0]["seq_name"],batch[0]["sequence"]
    seq_data = [(name, seq) for name,seq in zip(names,seqs)]
    batch_labels, batch_strs, batch_tokens = batch_converter(seq_data)
    
    # 2. Get filenames
    filenames = [item["filename"] for item in batch]  # Extract filenames from batch

    return {
        "batch_labels": batch_labels,
        "batch_strs": batch_strs,
        "batch_tokens": batch_tokens,
        "filenames": filenames  # Add filenames to batch_data
    }

# Wrapper to pass dataset.batch_converter into collate_fn
def collate_fn_wrapper(batch, dataset):
    return custom_collate_fn(batch, dataset.batch_converter)
old_vocab = {
    0: '<bos>',
    2: '<eos>',
    4: 'A',
    5: 'C',
    6: 'G',
    7: 'U',
}
vocab_dict = {
 'A': 1, 'G': 2, 'T': 3, 'C': 4, 'U': 5, 'N': 6, 'R': 7, 'Y': 8, 'K': 9,
 'M': 10, 'S': 11, 'W': 12, 'B': 13, 'D': 14, 'H': 15, 'V': 16, 'Z': 17,
 '.': 18, '-': 19,'<bos>':18,'<eos>':19
}

def predict(model, bert_model, linear, resnet, dataloader, output_dir, threshold=0.5, is_eval=True, device: torch.device = torch.device("cpu")):
    resnet.eval()
    linear.eval()
    bert_model.eval()
    run_dir = output_dir
    os.makedirs(run_dir, exist_ok=True)

    batch_idx = 0
    with torch.no_grad():
        for batch_data in dataloader:
            batch_idx += 1
            tokens = batch_data["batch_tokens"].to(device)
            new_tokens = torch.tensor([
                [vocab_dict.get(old_vocab.get(int(t), ''), 0) for t in seq]
                for seq in tokens
            ], device=device)
            embed = bert_model(
                input_ids=new_tokens.unsqueeze(0)
            )["hidden_states"].squeeze(0)
            embed = linear(embed)
            ds_module_input = {"embedding": embed}
            results = resnet(tokens, ds_module_input)
            preds = torch.sigmoid(results)
            seq = batch_data["batch_strs"][0]
            pred_prob = preds.mean(dim=0)
            test_name = batch_data["filenames"][0]
            file_path = os.path.join(run_dir, f"{test_name}.npz")
            np.savez_compressed(
                file_path,
                pred_prob=pred_prob.cpu().numpy(),
                batch_idx=batch_idx
            )
            print(f"[Prediction] Saved: {file_path} (test_name: {test_name})", flush=True)

    print(f"Prediction finished. Results saved in {run_dir}")
    return run_dir

if __name__=='__main__':

    parser = argparse.ArgumentParser(description="Predict secondary structure from a3m inputs")
    parser.add_argument("--input_a3m", default="../example_data", help="Directory containing .a3m files")
    parser.add_argument("--output_dir", default="../output_ss", help="Directory to save predictions")
    parser.add_argument("--device", default="cpu", help="Device to run on, e.g. cpu or cuda:0")
    parser.add_argument("--model_path", default="../pretrained/RMSA_ss.pt", help="Path to bundled model/config/weights")
    args = parser.parse_args()

    device = torch.device(args.device)
    SEED = 42  # You can change this to any integer value
    set_seed(SEED)
    a3m_dir = args.input_a3m

    # Load bundle
    bundle = torch.load(args.model_path, map_location=device)
    config_dict = bundle["config"]
    model_config = MSABertConfig(**config_dict)
    alphabet_config = model_config.alphabet_config
    vocabs = alphabet_config["standard_toks"]
    for i, w in enumerate(vocabs):
        vocab_dict[w] = i + 1

    # Initialize dataset
    dataset = RNADataLoader(a3m_dir)
    print(f"Test: {len(dataset)}")
    test_loader = DataLoader(dataset, batch_size=1, shuffle=False,
                            collate_fn=lambda batch: collate_fn_wrapper(batch, dataset))

    # Build model structure
    model, _ = fm.downstream.build_rnafm_resnet(type="ss")
    model = model.to(device)
    bert_model = MSABertForPretrain(model_config)
    bert_model.load_state_dict(bundle["bert_state_dict"])
    bert_model.eval().to(device)
    linear = nn.Linear(768, 640).to(device)
    linear.load_state_dict(bundle["linear_state_dict"])
    resnet = model.downstream_modules['pc-resnet_1_sym_first:r-ss']
    resnet.input_msa = True
    resnet.msa_embed_dim_in = resnet.embed_dim
    resnet.msa_embed_dim_out = resnet.embed_dim // resnet.attention_heads
    resnet.msa_q_proj = nn.Linear(resnet.msa_embed_dim_in, resnet.msa_embed_dim_out)
    resnet.msa_k_proj = nn.Linear(resnet.msa_embed_dim_in, resnet.msa_embed_dim_out)
    resnet.depth_reduction = 'mean'
    resnet = resnet.to(device)
    resnet.load_state_dict(bundle["resnet_state_dict"])

    # Prediction only (no training or validation)
    predict(model, bert_model, linear, resnet, test_loader, args.output_dir, threshold=0.5, device=device)