import numpy as np
import scipy.sparse as sp

from tools.tools import sparse_mx_to_torch_sparse_tensor, normalize_adj


def get_mp(drug_drug, durg_protein, drug_disease, drug_se, protein_protein, protein_disease, device):
    drdr = drug_drug
    drpr = durg_protein
    prpr = protein_protein

    drprdr = np.matmul(drpr, drpr.T) > 0
    prdrpr = np.matmul(drpr.T, drpr) > 0
    drprpr = np.matmul(drpr, prpr) > 0
    drprprdr = np.matmul(drprpr, drpr.T) > 0
    drprdr = np.array(drprdr)
    prdrpr = np.array(prdrpr)
    prdrdr = np.matmul(drpr.T, drdr) > 0
    prdrdrpr = np.matmul(prdrdr, drpr) > 0


    drdr = sparse_mx_to_torch_sparse_tensor(normalize_adj(sp.coo_matrix(drdr))).to(device)
    drprdr = sparse_mx_to_torch_sparse_tensor(normalize_adj(sp.coo_matrix(drprdr))).to(device)
    drprprdr = sparse_mx_to_torch_sparse_tensor(normalize_adj(sp.coo_matrix(drprprdr))).to(device)

    prpr = sparse_mx_to_torch_sparse_tensor(normalize_adj(sp.coo_matrix(prpr))).to(device)
    prdrpr = sparse_mx_to_torch_sparse_tensor(normalize_adj(sp.coo_matrix(prdrpr))).to(device)
    prdrdrpr = sparse_mx_to_torch_sparse_tensor(normalize_adj(sp.coo_matrix(prdrdrpr))).to(device)

    drug = 'drug'
    protein = 'protein'
    disease = 'disease'
    sideeffect = 'sideeffect'

    mps_dict = {drug: [drdr, drprdr, drprprdr], protein: [prpr, prdrpr, prdrdrpr], disease: [], sideeffect: []}

    return mps_dict
