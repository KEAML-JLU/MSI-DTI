# MSI-DTI
The source code of "MSI-DTI: Predicting Drug-Target Interaction Based on Multi-Source Information and Multi-Head Self-Attention"


## Datasets

Four datasets were used in the experiments: DTKG, Davis, KIBA and DrugBank, of which all the data of Davis and DrufBank as well as the example data of DTKG we put in the corresponding folders, and KIBA can be downloaded by yourself according to the links.

Before running, it is necessary to process the data, specifically, it is necessary to extract the drug information from all the data and save it into drugs.csv, the example is as follows:

drug_id,smiles  

11314340,CC1=C2C=C(C=CC2=NN1)C3=CC(=C...  
24889392,CC(C)(C)C1=CC(=NO1)NC(=O)NC2=CC=C(C=C2)C3=CN4C5=C(...  
11409972,CCN1CCN(CC1)CC2=C(C=C(C=C2)NC(=O)NC3=CC=C(C=C3)O...  
11338033,C1CNCCC1NC(=O)C2=C(C=NN2)NC(=O)C3=...  

Extract target information into proteins.csv, the example is as follows:  

0,pro_ids,seq  

AAK1,AAK1,MKKFFDSRREQGGSGLGSGSSGGGGSTSGLGSGYIGRVFGIGRQ...  
ABL1(E255K),ABL1(E255K),PFWKILNPLLERGTYYYFMGQQPGKVLGDQRR...  
AKT1,AKT1,MSDVAIVKEGWLHKRGEYIKTWRPRYFLLKNDGTFIGYKER...  
AKT2,AKT2,MNEVSVIKEGWLHKRGEYIKTWRPRYFLLKSDGSFIGYKERPEAPDQTLPPLNN...  

## Requirements

python == 3.7.16  
tensorflow == 1.15.0  
deepctr == 0.8.4  
frufs == 1.0.2  
keras == 2.11.0  
numpy == 1.18.4  
pandas == 1.1.5  
scikit-learn == 0.24.1  
scipy == 1.7.3

## Usage

Firstly, the drug representation of the selected dataset is obtained by running smiles.py  
```
python smiles.py
```
then seq.py is run to obtain the target representation of the selected dataset  
```
python seq.py
```
it is worth mentioning that, in the network embedding module, we have used both [AttentionWalk](https://github.com/benedekrozemberczki/AttentionWalk "AttentionWalk") and [CompGCN](https://github.com/malllabiisc/CompGCN "CompGCN") methods, and their running methods are shown in the source file.  

After preparing these materials, run the main.py(we take the KIBA dataset as an example)  
```
python main.py
```

