import numpy as np
import scipy.sparse as sp
from tools.tools import load_data, sparse_mx_to_torch_sparse_tensor
from scipy.sparse import coo_matrix

drug = 'drug'
protein = 'protein'
disease = 'disease'
sideeffect = 'sideeffect'

DR_PR_I = 'drug_protein interaction'
PR_DR_I = 'protein_drug interaction'

DR_DR_A = 'drug_drug association'
DR_PR_A = 'drug_protein association'
DR_DI_A = 'drug_disease association'
DR_SE_A = 'drug_sideeffect association'

PR_DR_A = 'protein_drug association'
PR_PR_A = 'protein_protein association'
PR_DI_A = 'protein_disease association'
PR_SE_A = 'protein_sideeffect association'

DI_DR_A = 'disease_drug association'
DI_PR_A = 'disease_protein association'
DI_DI_A = 'disease_disease association'
DI_SE_A = 'disease_sideeffect association'

SE_DR_A = 'sideeffect_drug association'
SE_PR_A = 'sideeffect_protein association'
SE_DI_A = 'sideeffect_disease association'
SE_SE_A = 'sideeffect_sideeffect association'

drug_pos_num = 10
protein_pos_num = 5
sideeffect_pos_num = 20
disease_pos_num = 20

drug_num = 708
protein_num = 1512
sideeffect_num = 4192
disease_num = 5603

drug_drug, drug_chemical, drug_disease, drug_sideeffect, protein_protein, \
protein_sequence, protein_disease, dti_original = load_data()


def getMetaPathSrcAndDst(g, metapath):
    adj = 1
    for etype in metapath:
        adj = adj * g.adj(etype=etype, scipy_fmt='csr', transpose=False)
    adj = adj.tocoo()
    return adj


def generate_pos(node_all, node_pos, node_pos_num):
    for i in range(len(node_all)):
        one = node_all[i].nonzero()[0]
        if len(one) > node_pos_num:
            oo = np.argsort(-node_all[i, one])
            sele = one[oo[:node_pos_num]]
            node_pos[i, sele] = 1
        else:
            node_pos[i, one] = 1
    return sp.coo_matrix(node_pos)


def get_pos(g, device):
    drdr = getMetaPathSrcAndDst(g, [DR_DR_A])
    drprdr = getMetaPathSrcAndDst(g, [DR_PR_I, PR_DR_I])
    drprprdr = getMetaPathSrcAndDst(g, [DR_PR_I, PR_PR_A, PR_DR_I])
    drdr = drdr / (drdr.sum(axis=-1) + 1e-12).reshape(-1, 1)
    drprdr = drprdr / (drprdr.sum(axis=-1) + 1e-12).reshape(-1, 1)
    drprprdr = drprprdr / (drprprdr.sum(axis=-1) + 1e-12).reshape(-1, 1)
    drug_all = (drdr + drprdr + drprprdr + coo_matrix(np.identity(drug_num))).A.astype("float32")
    drug_pos = np.zeros((drug_num, drug_num))
    drug_pos = generate_pos(drug_all, drug_pos, drug_pos_num)

    # protein
    prpr = getMetaPathSrcAndDst(g, [PR_PR_A])
    prdrpr = getMetaPathSrcAndDst(g, [PR_DR_I, DR_PR_I])
    prdrdrpr = getMetaPathSrcAndDst(g, [PR_DR_I, DR_DR_A, DR_PR_I])
    prpr = prpr / (prpr.sum(axis=-1) + 1e-12).reshape(-1, 1)
    prdrpr = prdrpr / (prdrpr.sum(axis=-1) + 1e-12).reshape(-1, 1)
    prdrdrpr = prdrdrpr / (prdrdrpr.sum(axis=-1) + 1e-12).reshape(-1, 1)
    protein_all = (prpr + prdrpr + prdrdrpr + coo_matrix(np.identity(protein_num))).A.astype("float32")
    protein_pos = np.zeros((protein_num, protein_num))
    protein_pos = generate_pos(protein_all, protein_pos, protein_pos_num)

    pos_dict = {drug: sparse_mx_to_torch_sparse_tensor(drug_pos).to(device),
                protein: sparse_mx_to_torch_sparse_tensor(protein_pos).to(device)}
    return pos_dict
