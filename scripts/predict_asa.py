import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from msa_transformer import MSABertForPretrain,MSABertConfig
import numpy as np
MAX_SEQ_LEN = 512
MAX_LINE_SIZE = 128
BATCH_SIZE = 4
vocab_dict = {}
class ChunkedDataset(Dataset):
    def __init__(self, original_dataset, pad_token_id=0):
        self.original_dataset = original_dataset
        self.pad_token_id = pad_token_id
        self.chunk_size = 64
        self.index_map = []

        # Build an index map to know which (d1, chunk_idx) to pick
        for dataset_idx, d1 in enumerate(original_dataset):
            n, seq_len = d1['input_ids'].shape
            remainder = n % self.chunk_size
            padded_n = n if remainder == 0 else n + (self.chunk_size - remainder)
            m = padded_n // self.chunk_size

            for chunk_idx in range(m):
                self.index_map.append((dataset_idx, chunk_idx))

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        dataset_idx, chunk_idx = self.index_map[idx]
        d1 = self.original_dataset[dataset_idx]

        input_ids = d1['input_ids']     # (n, 512)

        n, seq_len = input_ids.shape
        remainder = n % self.chunk_size
        if remainder != 0:
            pad_len = self.chunk_size - remainder
            pad_tensor = torch.full((pad_len, seq_len), self.pad_token_id, dtype=input_ids.dtype)
            input_ids = torch.cat([input_ids, pad_tensor], dim=0)

        # Now split into chunks
        input_ids = input_ids.view(-1, self.chunk_size, seq_len)  # (m, 128, 512)

        return {
            'input_ids': input_ids[chunk_idx],          # (128, 512)
            'f': d1['f'],
            'chunk_idx': chunk_idx
        }

class CNNTokenClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super(CNNTokenClassifier, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.fc = nn.Linear(hidden_dim, 1)  # project per token to 1 score
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
    def forward(self, x):
        # x shape: [batch_size, seq_len, input_dim]
        x = x.transpose(1, 2)  # -> [batch_size, input_dim, seq_len]
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.transpose(1, 2)  # -> [batch_size, seq_len, hidden_dim]
        x = self.dropout(x)
        x = self.fc(x)  # -> [batch_size, seq_len, 1]
        x = x.squeeze(-1)  # -> [batch_size, seq_len]
        return x
class RNADataset(Dataset):
    def __init__(self, data_list, vocab_dict = vocab_dict):
        self.data_list = data_list
        self.vocab_dict = vocab_dict
        
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index):
        # DEBUG
        file = self.data_list[index][0]
        f = open(file, "r")
        input_ids = []
        for line in f.readlines()[:64]:
            line_ids = []
            if not line.startswith(">"):
                txt = line.strip()
                for t in txt:
                    line_ids.append(self.vocab_dict[t])
                    if len(line_ids) > MAX_SEQ_LEN:
                        line_ids = line_ids[:MAX_SEQ_LEN]
                    else:
                        line_ids.extend([0] * (MAX_SEQ_LEN - len(line_ids)))
                # line_ids_tensor = torch.LongTensor(line_ids)
                line_ids_tensor = torch.LongTensor(line_ids)
                input_ids.append(line_ids_tensor)
        while (len(input_ids) < MAX_LINE_SIZE):
            input_ids.append(torch.LongTensor([0] * MAX_SEQ_LEN))
        input_ids = torch.stack(input_ids, dim = 0)
        return {
            'input_ids': input_ids,
            'f': file
        }

def load_bundle_and_setup_vocab(bundle_path):
    bundle = torch.load(bundle_path, map_location="cpu")
    config_dict = bundle["config"]
    model_config = MSABertConfig(**config_dict)
    alphabet_config = model_config.alphabet_config
    vocabs = alphabet_config["standard_toks"]
    for i, w in enumerate(vocabs):
        vocab_dict[w] = i + 1
    return bundle, model_config

def main():

    parser = argparse.ArgumentParser(description="Predict ASA from a3m inputs")
    parser.add_argument("--input_a3m", default="../example_data", help="Directory containing .a3m files")
    parser.add_argument("--output_dir", default="../output_asa", help="Directory to save predictions")
    parser.add_argument("--device", default="cpu", help="Device to run on, e.g. cpu or cuda:0")
    parser.add_argument("--model_path", default="../pretrained/RMSA_asa.pt", help="Path to bundled model/config/weights")
    args = parser.parse_args()

    bundle, model_config = load_bundle_and_setup_vocab(args.model_path)

    input_file_list = [
        os.path.join(args.input_a3m, f)
        for f in os.listdir(args.input_a3m)
        if f.endswith(".a3m")
    ]
    print(len(input_file_list))

    # Prepare dataset
    dataset = RNADataset([(f, None) for f in input_file_list])
    dataset = ChunkedDataset(dataset)

    # Create DataLoader
    test_loader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # Load BERT model
    bert_model = MSABertForPretrain(model_config)
    bert_model.load_state_dict(bundle["bert_state_dict"])
    bert_model.eval()

    # Load CNN model
    cnn_model = CNNTokenClassifier(input_dim=768, hidden_dim=256)
    cnn_model.load_state_dict(bundle["cnn_state_dict"])
    cnn_model.eval()

    # Set device
    device = args.device
    bert_model = bert_model.to(device)
    cnn_model = cnn_model.to(device)

    # Define loss function
    criterion = nn.MSELoss()

    # Inference
    cnn_model.eval()
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)


    # Aggregate predictions for each filename
    pred_dict = {}
    with torch.no_grad():
        for batch in tqdm(test_loader, total=len(test_loader), desc="Evaluating:"):
            input_ids = batch['input_ids'].to(device)
            filenames = [os.path.splitext(os.path.basename(f))[0] for f in batch['f']]
            chunk_indices = batch['chunk_idx']

            # Get BERT features
            bert_outputs = bert_model(input_ids=input_ids)
            features = bert_outputs["hidden_states"]
            features = features.mean(dim=1)

            # CNN prediction
            outputs = cnn_model(features)
            predictions = outputs.cpu().numpy()

            for i, filename in enumerate(filenames):
                chunk_idx = int(chunk_indices[i])
                if filename not in pred_dict:
                    pred_dict[filename] = {}
                pred_dict[filename][chunk_idx] = predictions[i]

    # Save all chunks for each filename into a single .npz file
    for filename, chunk_map in pred_dict.items():
        # Sort by chunk_idx to ensure correct order
        sorted_chunks = [chunk_map[idx] for idx in sorted(chunk_map.keys())]
        all_preds = np.stack(sorted_chunks, axis=0)  # shape: (num_chunks, chunk_size, seq_len)
        output_path = os.path.join(output_dir, f"{filename}.npz")
        np.savez_compressed(output_path, pred=all_preds)

    print(f"Results saved to {output_dir}")

if __name__ == "__main__":
    main()


