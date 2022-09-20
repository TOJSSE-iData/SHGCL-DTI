# SHGCL-DTI

## Requirements

* python 3.9.7
* cuda 10.2
* pytorch 1.11.0
* dgl 0.8.0
* numpy 1.21.2
* sklearn 1.0.2

## Quick start

1. cd src/application
2. python `main.py` Options are:
   1. `--device: the gpu used to train model, default:cuda:0`
   2. `--hid_dim: the dimension of hidden layers, default:2048`
   3. `--number: the Postive and Negative ratio, three choice: one--pos:neg=1:1; ten--pos:neg=1:10; all--all unlabelled DTI is treated as neg, default:ten`
   4. `--feature: the feature used in experiment, three choice: random, luo, default(ours), default:default`
   5. `--task: the different scenario to test: choice: benchmark->mat_drug_protein.txt, disease->mat_drug_protein_disease.txt, drug->mat_drug_protein_drug.txt, homo_protein_drug->mat_drug_protein_homo_protein_drug.txt,sideeffect->mat_drug_protein_sideeffect.txt, unique->mat_drug_protein_drug_unique.txt default:benchmark`
   6. `--edge_mask: whether mask some edges in the HN, choice: ''(empty str) drug protein drug,protein disease sideeffect disease,sideeffect drugsim proteinsim drugsim,proteinsim default:''(empty str)'`


## Data description

* `drug.txt` : list of drug names.
* `protein.txt` : list of protein names.
* `disease.txt` : list of disease names.
* `se.txt` : list of side effect names.
* `drug_dict_map.txt` : a complete ID mapping between drug names and DrugBank ID.
* `protein_dict_map.txt`: a complete ID mapping between protein names and UniProt ID.
* `mat_drug_se.txt` : Drug-SideEffect association matrix.
* `mat_protein_protein.txt` : Protein-Protein interaction matrix.
* `mat_drug_drug.txt` : Drug-Drug interaction matrix.
* `mat_protein_disease.txt` : Protein-Disease association matrix.
* `mat_drug_disease.txt` : Drug-Disease association matrix.
* `mat_protein_drug.txt` : Protein-Drug interaction matrix.
* `mat_drug_protein.txt` : Drug-Protein interaction matrix.
* `Similarity_Matrix_Drugs.txt` : Drug similarity scores based on chemical structures of drugs
* `Similarity_Matrix_Proteins.txt` : Protein similarity scores based on primary sequences of proteins
* `mat_drug_protein_homo_protein_drug.txt` : Drug-Protein interaction matrix, in which DTIs with similar drugs (i.e., drug chemical structure similarities > 0.6) or similar proteins (i.e., protein sequence similarities > 40%) were removed (see the paper). 
* `mat_drug_protein_drug.txt` : Drug-Protein interaction matrix, in which DTIs with drugs sharing similar drug interactions (i.e., Jaccard similarities > 0.6) were removed (see the paper). 
* `mat_drug_protein_sideeffect.txt` : Drug-Protein interaction matrix, in which DTIs with drugs sharing similar side effects (i.e., Jaccard similarities > 0.6) were removed (see the paper). 
* `mat_drug_protein_disease.txt` : Drug-Protein interaction matrix, in which DTIs with drugs or proteins sharing similar diseases (i.e., Jaccard similarities > 0.6) were removed (see the paper). 
* `mat_drug_protein_unique.txt` : Drug-Protein interaction matrix, in which known unique and non-unique DTIs were labelled as 3 and 1, respectively, the corresponding unknown ones were labelled as 2 and 0 (see the paper for the definition of unique). 

These files: drug.txt, protein.txt, disease.txt, se.txt, drug_dict_map, protein_dict_map, mat_drug_se.txt, mat_protein_protein.txt, mat_drug_drug.txt, mat_protein_disease.txt, mat_drug_disease.txt, mat_protein_drug.txt, mat_drug_protein.txt, Similarity_Matrix_Proteins.txt, are extracted from https://github.com/luoyunan/DTINet.

These files: mat_drug_protein_homo_protein_drug.txt, mat_drug_protein_drug.txt, mat_drug_protein_sideeffect.txt, mat_drug_protein_disease.txt, mat_drug_protein_unique.txt are extracted from https://github.com/FangpingWan/NeoDTI