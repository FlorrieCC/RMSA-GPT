import os
import sys
import argparse
sys.path.append('/home/coop/jyjiang/Projects/MSA/supp_0101/MSAGPT') 
import json
import tempfile
import torch
from transformers import GPT2Tokenizer, GPT2Config
from esm import Alphabet
from msa_transformer import MSABertConfig, MSATransformerWithGPT2Config, MSATransformerWithGPT2

# Model loading function
def load_model_bundle(model_path, device):
    bundle = torch.load(model_path, map_location="cpu", weights_only=True)

    # Load BERT config
    alphabet_config = bundle["alphabet_config"]
    alphabet = Alphabet(**alphabet_config)
    batch_converter = alphabet.get_batch_converter()

    # Load MSA model
    config_dict = bundle["msa_config"]
    msa_bert_config = MSABertConfig(**config_dict["msa_bert_config"])
    gpt2_config = GPT2Config(**config_dict["gpt2_config"])
    msa_config = MSATransformerWithGPT2Config(
        msa_bert_config=msa_bert_config,
        gpt2_config=gpt2_config,
    )
    msa_transformer = MSATransformerWithGPT2(msa_config)
    msa_transformer.load_state_dict(bundle["state_dict"])
    msa_transformer = msa_transformer.to(device).eval()

    # Load GPT2 tokenizer
    tmp_dir = tempfile.TemporaryDirectory()
    vocab_path = os.path.join(tmp_dir.name, "vocab_dotseq.json")
    merges_path = os.path.join(tmp_dir.name, "merges_single.txt")
    with open(vocab_path, "w") as f:
        f.write(bundle["tokenizer"]["vocab_json"])
    with open(merges_path, "w") as f:
        f.write(bundle["tokenizer"]["merges_txt"])
    gpt2_tokenizer = GPT2Tokenizer(vocab_file=vocab_path, merges_file=merges_path)

    # Return all objects and temp dir (to keep files alive)
    return msa_transformer, gpt2_tokenizer, batch_converter, tmp_dir

# Parse .fasta files
def process_fasta_file(file_path):
    sequences = []
    with open(file_path, 'r') as f:
        sequence = ""
        for line in f:
            if line.startswith(">"):  # Header line
                if sequence:  # Save accumulated sequence
                    sequences.append(sequence)
                sequence = ""  # Reset current sequence
            else:
                sequence += line.strip()  # Accumulate sequence
        if sequence:  # Save last sequence
            sequences.append(sequence)
    return sequences

def generate_for_file(file_name):
    file_path = os.path.join(msa_U_dir, file_name)
    output_fasta_path = os.path.join(generate_output_dir, file_name)

    # Skip if existing FASTA has >= 400 lines
    if os.path.exists(output_fasta_path):
        with open(output_fasta_path, 'r') as existing_fasta:
            line_count = sum(1 for _ in existing_fasta)
            if line_count >= 400:
                print(f"Skipping {file_name} as it already exists with >= 400 lines.")
                return

    print(f"Processing {file_path}...")

    # Read all sequences from FASTA
    sequences = process_fasta_file(file_path)

    # Use all sequences as one input
    data = [(f"seq{i+1}", seq) for i, seq in enumerate(sequences)]  # Format: ("seq1", "A-GAGGAATACCCATG")
    batch_labels, batch_strs, batch_tokens = batch_converter(data)

    # Move inputs to device
    batch_tokens = batch_tokens.to(device)

    # Ensure token ids are within vocab range
    if any(token >= gpt2_tokenizer.vocab_size for token in batch_tokens.flatten().tolist()):
        print(f"Warning: token exceeds vocab size for {file_name}")
        return

    # Generate sequences
    generated_sequences = []
    with torch.inference_mode():
        for i in range(20):
            o = msa_transformer.generate(input_ids=batch_tokens)
            result = gpt2_tokenizer.decode(o[0].tolist())
            generated_sequences.append(result)

            # Log output
            print(f"Generated sequence {i+1} for {file_name}: {result}")

    # Save generated sequences to a new FASTA file
    with open(output_fasta_path, 'w') as fasta_file:
        for i, seq in enumerate(generated_sequences):
            # Replace T->U, filter non-AUGC, and uppercase
            cleaned_seq = ''.join([char.upper() for char in seq.replace('T', 'U') if char.upper() in 'AUGC'])
            fasta_file.write(f">{file_name.replace('.fasta', '')}_seq{i+1}\n")
            fasta_file.write(f"{cleaned_seq}\n")

if __name__ == "__main__":


    parser = argparse.ArgumentParser(description="Generate sequences from MSA FASTA files")
    parser.add_argument("--input_dir", default='../example_data', help="Directory containing .fasta files")
    parser.add_argument("--output_dir", default='../output_seq', help="Directory to save generated FASTA files")
    parser.add_argument("--device", default="cpu", help="Device to run on, e.g. cpu or cuda:0")
    parser.add_argument("--model_path", default='../pretrained/RMSA_gen.pt', help="Path to the bundled model file (.pt)")
    args = parser.parse_args()

    msa_U_dir = args.input_dir
    generate_output_dir = args.output_dir
    device = torch.device(args.device)
    model_path = args.model_path

    # Load model and tokenizer from bundle
    msa_transformer, gpt2_tokenizer, batch_converter, tmp_dir = load_model_bundle(model_path, device)

    # Ensure output directory exists
    if not os.path.exists(generate_output_dir):
        os.makedirs(generate_output_dir)

    # Collect FASTA files
    fasta_files = [file_name for file_name in os.listdir(msa_U_dir) if file_name.endswith('.fasta')]

    for file_name in fasta_files:
        generate_for_file(file_name)

    print("Done.")
