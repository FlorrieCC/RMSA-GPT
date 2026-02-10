# RMSA-GPT

## Environment Setup
  
**After configuring the environment, you must download the required model files from:**

https://mycuhk-my.sharepoint.com/personal/1155210737_link_cuhk_edu_hk/_layouts/15/onedrive.aspx?id=%2Fpersonal%2F1155210737%5Flink%5Fcuhk%5Fedu%5Fhk%2FDocuments%2Fall%5Fnature%5Fdata%2FRMSA%2DGPT%2Fmodel%5Ffile&viewid=2494af49%2D1967%2D4bf8%2Da92f%2Dfcc2556b6a55

Download the following files and place them in the `pretrained` directory:

- **ASA Prediction:** `RMSA_asa.pt`
- **Secondary Structure Prediction:** `RMSA_ss.pt`
- **Sequence Generation Task:** `RMSA_gen.pt`

1. Create a new conda environment (recommended Python 3.10):

```bash
conda create -n rmsa-gpt python=3.10
```

2. Activate the environment:

```bash
conda activate rmsa-gpt
```

3. Install all dependencies using the provided environment.yaml:

```bash
conda env update -f environment.yaml
```

---

## Project Structure & Example Data

- The project is divided into two main tasks: **Prediction** and **Generation**.
- Example input data is provided in the `example_data` directory:
	- Files ending with `.a3m` are used for prediction tasks (ASA and secondary structure).
	- Files ending with `.fasta` are used for sequence generation tasks.

---

## 1. Prediction Tasks

### (A) ASA Prediction

1. Enter the scripts directory:

```bash
cd RMSA-GPT/scripts
```

2. Run ASA prediction (all command-line arguments shown with their default values):

```bash
python predict_asa.py \
	--input_a3m ../example_data \
	--output_dir ../output_asa \
	--device cpu \
	--model_path ../pretrained/RMSA_asa.pt
```

- **Input:** All `.a3m` files in `../example_data`.
- **Output:** For each input, a `.npz` file will be saved in `../output_asa`, containing the predicted ASA values for all sequence chunks.

### (B) Secondary Structure Prediction

1. Enter the scripts directory:

```bash
cd RMSA-GPT/scripts
```

2. Run secondary structure prediction (all command-line arguments shown with their default values):

```bash
python predict_secondary_structure.py \
	--input_a3m ../example_data \
	--output_dir ../output_ss \
	--device cpu \
	--model_path ../pretrained/RMSA_ss.pt
```

- **Input:** All `.a3m` files in `../example_data`.
- **Output:** For each input, a `.npz` file will be saved in `../output_ss`, containing the predicted secondary structure probabilities.

---

## 2. Sequence Generation Task

- Both family and RBP-binding RNA sequence generation use the same script; only the input data differs.
- Example input data is provided in `example_data` (files ending with `.fasta`).

1. Enter the scripts directory:

```bash
cd RMSA-GPT/scripts
```

2. Run sequence generation (all command-line arguments shown with their default values):

```bash
python generate_seq.py \
	--input_dir ../example_data \
	--output_dir ../output_seq \
	--device cpu \
	--model_path ../pretrained/RMSA_gen.pt
```

- **Input:** All `.fasta` files in `../example_data`.
- **Output:** For each input, a new FASTA file will be saved in `../output_seq`, containing the generated RNA sequences.

---

## Notes
- All scripts support `--device` (e.g., `cpu` or `cuda:0`).
- Make sure the required model bundle files are present in the `../pretrained` directory.
- Output directories will be created automatically if they do not exist.
