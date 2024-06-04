import os
import argparse
import pandas as pd
from rdkit import Chem
import networkx as nx
from karateclub import Graph2Vec
import deepchem as dc
import numpy as np
from generate_fp_and_descriptors import Features_Generations

def generate_fp(input_file, output_folder):
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)

    fg = Features_Generations(input_file)
    
    fingerprint_types = ['morgan_fp', 'maccs_fp', 'avalon_fp']
    output_files = ['morgan.csv', 'maccs.csv', 'avalon.csv']
    
    for idx, fingerprint_type in enumerate(fingerprint_types):
        if fingerprint_type == 'morgan_fp':
            numpy_file = fg.morgan_fp()
        elif fingerprint_type == 'maccs_fp':
            numpy_file = fg.maccs_fp()
        elif fingerprint_type == 'avalon_fp':
            numpy_file = fg.avalon_fp()
        else:
            print(f'Invalid fingerprint type: {fingerprint_type}')
            continue
        
        output_file = os.path.join(output_folder, output_files[idx])
        np.savetxt(output_file, numpy_file)
        print(f'{fingerprint_type} fingerprints saved to {output_file}')

if __name__ == "__main__":
    # Choose the dataset (DTKG, Davis, KIBA, DrugBank)
    dataset = "KIBA"
    
    input_file = f'./{dataset}/drugs.csv'
    output_folder = f'./{dataset}/smiles'

    # Read the data file
    print(">>> Reading the data file ...")
    mol = pd.read_csv(input_file)
    print(">>> Data shape =", mol.shape)
    print(">>> Data columns =", mol.columns, "\n")
    print(mol)
    print()

    # Create RDKit molecule objects from SMILES
    print(">>> Creating molecules from SMILES ...")
    mol['mol'] = mol['smiles'].apply(lambda x: Chem.MolFromSmiles(x))

    # Define function to convert RDKit molecule to NetworkX graph
    def mol_to_nx(mol):
        G = nx.Graph()
        for atom in mol.GetAtoms():
            G.add_node(atom.GetIdx(),
                       atomic_num=atom.GetAtomicNum(),
                       is_aromatic=atom.GetIsAromatic(),
                       atom_symbol=atom.GetSymbol())
        for bond in mol.GetBonds():
            G.add_edge(bond.GetBeginAtomIdx(),
                       bond.GetEndAtomIdx(),
                       bond_type=bond.GetBondType())
        return G

    # Create NetworkX graphs from RDKit molecules
    print(">>> Creating NetworkX graphs from molecules ...")
    mol['graph'] = mol['mol'].apply(lambda x: mol_to_nx(x))

    # Graph2Vec feature extraction
    print(">>> Creating graph embeddings (Graph2Vec) ...")
    model = Graph2Vec()
    model.fit(mol['graph'])
    mol_graph2vec = model.get_embedding()

    mol_graph2vec_df = pd.DataFrame(mol_graph2vec)
    mol_graph2vec_df.to_csv(f'{output_folder}/graph2vec.csv', index=False, header=False)
    print(">>> Saved Graph2Vec features to graph2vec.csv")

    # Mol2Vec feature extraction
    print(">>> Creating molecule embeddings (Mol2Vec) ...")
    smiles = mol['smiles'].tolist()
    featurizer = dc.feat.Mol2VecFingerprint()
    features = featurizer.featurize(smiles)

    features_df = pd.DataFrame(features)
    features_df.to_csv(f'{output_folder}/mol2vec.csv', index=False, header=False)
    print(">>> Saved Mol2Vec features to mol2vec.csv")

    # Fingerprints extraction
    generate_fp(input_file, output_folder)
