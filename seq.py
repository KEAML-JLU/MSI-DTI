import torch
from CPCProt.tokenizer import Tokenizer
from CPCProt import CPCProtModel, CPCProtEmbedding
import pandas as pd
import os
from ifeatpro.features import get_feature, get_all_features

os.environ['CUDA_VISIBLE_DEVICES'] = '7'
device = "cpu"

# Initialize CPCProtModel and related components
model = CPCProtModel().to(device)
embedder = CPCProtEmbedding(model)
tokenizer = Tokenizer()

data = pd.read_csv('proteins.csv')
sequences = data['seq']

# Choose the dataset (DTKG, Davis, KIBA, DrugBank)
dataset = "KIBA"

# Processing sequences and saving features
fasta_file = f"proteins_{dataset.lower()}.fasta"  # Adjusted fasta file name based on dataset

output_dir = f"./{dataset}/seq/"  # Output directory based on dataset

for seq in sequences:
    input = torch.tensor([tokenizer.encode(seq)])   # (1, L)

    z_mean = embedder.get_z_mean(input)   # (1, 512)
    z_mean = z_mean.detach().cpu().numpy()
    z_mean = pd.DataFrame(z_mean)
    z_mean.to_csv(f'./{output_dir}/cpcport.csv', index=None, header=None, mode='a')

# Extracting features using ifeatpro
get_feature(fasta_file, "geary", output_dir)
get_feature(fasta_file, "ctdc", output_dir)
get_feature(fasta_file, "ctdt", output_dir)
get_feature(fasta_file, "ctdd", output_dir)
get_feature(fasta_file, "qsorder", output_dir)
get_feature(fasta_file, "paac", output_dir)

print("Feature extraction and sequence processing complete.")
