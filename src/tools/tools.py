from random import random

import numpy
import numpy as np
import dgl
import scipy.sparse as sp
from dgl.heterograph import DGLHeteroGraph
import torch
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from scipy.sparse import coo_matrix
import torch as th
import sys

sys.path.append('../')
from tools.args import parse_args

args = parse_args()
device = args.device

DR_DR_V = 'drug_drug virtual'
PR_PR_V = 'protein_protein virtual'
DI_DI_V = 'disease_disease virtual'
SE_SE_V = 'sideeffect_sideeffect virtual'

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

drug = 'drug'
protein = 'protein'
disease = 'disease'
sideeffect = 'sideeffect'

# 在异构转同构的过程中,num_nodes_per_ntype为[5603,708,1512,4192] 分别对应di drug pro se
drug_len = 708
protein_len = 1512
sideeffect_len = 4192
disease_len = 5603


def saveTxt(features: list[str], path: str):
    with open(path, "w") as f:
        for feature in features:
            f.write(feature + '\n')
    print("txt save finished")


def load_data(dti_path='mat_drug_protein.txt'):
    network_path = '../../data/data/'
    true_drug = drug_len

    drug_drug = np.loadtxt(network_path + 'mat_drug_drug.txt')
    drug_chemical = np.loadtxt(network_path + 'Similarity_Matrix_Drugs.txt')
    drug_chemical = drug_chemical[:true_drug, :true_drug]
    drug_disease = np.loadtxt(network_path + 'mat_drug_disease.txt')
    drug_sideeffect = np.loadtxt(network_path + 'mat_drug_se.txt')

    protein_protein = np.loadtxt(network_path + 'mat_protein_protein.txt')
    protein_sequence = np.loadtxt(network_path + 'Similarity_Matrix_Proteins.txt')
    protein_disease = np.loadtxt(network_path + 'mat_protein_disease.txt')

    num_drug = len(drug_drug)
    num_protein = len(protein_protein)

    # Removed the self-loop
    drug_chemical = drug_chemical - np.identity(num_drug)
    protein_sequence = protein_sequence / 100.
    protein_sequence = protein_sequence - np.identity(num_protein)

    drug_protein = np.loadtxt(network_path + dti_path)

    edge_mask = args.edge_mask
    if edge_mask == 'drug':
        drug_drug[drug_drug != 0] = 0
    elif edge_mask == 'protein':
        protein_protein[protein_protein != 0] = 0
    elif edge_mask == 'drug,protein':
        drug_drug[drug_drug != 0] = 0
        protein_protein[protein_protein != 0] = 0
    elif edge_mask == 'disease':
        drug_disease[drug_disease != 0] = 0
        protein_disease[protein_disease != 0] = 0
    elif edge_mask == 'sideeffect':
        drug_sideeffect[drug_sideeffect != 0] = 0
    elif edge_mask == 'disease,sideeffect':
        drug_disease[drug_disease != 0] = 0
        protein_disease[protein_disease != 0] = 0
        drug_sideeffect[drug_sideeffect != 0] = 0
    elif edge_mask == 'drugsim':
        drug_chemical[drug_chemical != 0] = 0
    elif edge_mask == 'proteinsim':
        protein_sequence[protein_sequence != 0] = 0
    elif edge_mask == 'drugsim,proteinsim':
        drug_chemical[drug_chemical != 0] = 0
        protein_sequence[protein_sequence != 0] = 0

    return drug_drug, drug_chemical, drug_disease, drug_sideeffect, protein_protein, protein_sequence, \
           protein_disease, drug_protein


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def load_feature(choice='default'):
    if choice == 'default':
        return load_feature_default()
    elif choice == 'random':
        return load_feature_random()
    elif choice == "luo":
        return load_feature_luo()
    else:
        return load_feature_default()


def load_feature_default():
    disease_feats = torch.from_numpy(
        numpy.loadtxt("../../data/feature/disease_feature.txt")).to(torch.float32).to(device)
    drug_feats = torch.from_numpy(
        numpy.loadtxt("../../data/feature/drug_feature.txt")).to(torch.float32).to(device)
    protein_feats = torch.from_numpy(
        numpy.loadtxt("../../data/feature/protein_feature.txt")).to(torch.float32).to(device)
    sideeffect_feats = torch.from_numpy(
        numpy.loadtxt("../../data/feature/sideeffect_feature.txt")).to(torch.float32).to(device)
    node_features = {drug: drug_feats, protein: protein_feats, disease: disease_feats,
                     sideeffect: sideeffect_feats}
    feat_dim = {drug: 128, protein: 128, disease: 128, sideeffect: 128}
    return node_features, feat_dim


def load_feature_random():
    disease_feats = torch.randn(5603, 128).to(device)
    drug_feats = torch.randn(708, 128).to(device)
    protein_feats = torch.randn(1512, 128).to(device)
    sideeffect_feats = torch.randn(4192, 128).to(device)
    node_features = {drug: drug_feats, protein: protein_feats, disease: disease_feats,
                     sideeffect: sideeffect_feats}
    feat_dim = {drug: 128, protein: 128, disease: 128, sideeffect: 128}
    return node_features, feat_dim


def load_feature_luo():
    disease_feats = torch.randn(5603, 128).to(device)
    drug_feats = torch.from_numpy(
        numpy.loadtxt("../../data/feature/drug_vector_d100.txt")).to(torch.float32).to(device)
    protein_feats = torch.from_numpy(
        numpy.loadtxt("../../data/feature/protein_vector_d400.txt")).to(torch.float32).to(device)
    sideeffect_feats = torch.randn(4192, 128).to(device)
    node_features = {drug: drug_feats, protein: protein_feats, disease: disease_feats,
                     sideeffect: sideeffect_feats}
    feat_dim = {drug: 100, protein: 400, disease: 128, sideeffect: 128}
    return node_features, feat_dim


def ConstructGraph(drug_drug, drug_chemical, drug_disease, drug_sideeffect, protein_protein, protein_sequence,
                   protein_disease, drug_protein, args=None, CO=True) -> DGLHeteroGraph:
    num_drug = len(drug_drug)
    num_protein = len(protein_protein)
    num_disease = len(drug_disease.T)
    num_sideeffect = len(drug_sideeffect.T)

    list_drug = [(i, i) for i in range(num_drug)]
    list_protein = [(i, i) for i in range(num_protein)]
    list_sideeffect = [(i, i) for i in range(num_sideeffect)]
    list_disease = [(i, i) for i in range(num_disease)]

    drug_chemical = np.array(drug_chemical)
    drug_chemical[drug_chemical < 0.4] = 0
    # drug_drug = coo_matrix(drug_drug + drug_chemical)
    drug_drug = coo_matrix(drug_drug)
    list_DDI = (drug_drug.row, drug_drug.col)

    protein_sequence = np.array(protein_sequence)
    protein_sequence[protein_sequence < 0.6] = 0
    # protein_protein = coo_matrix(protein_protein + protein_sequence)
    protein_protein = coo_matrix(protein_protein)
    list_PPI = (protein_protein.row, protein_protein.col)

    list_SESEI = []

    list_DIDII = []

    drug_protein = coo_matrix(drug_protein)
    list_drug_protein = (drug_protein.row, drug_protein.col)
    list_protein_drug = (drug_protein.col, drug_protein.row)

    drug_sideeffect = coo_matrix(drug_sideeffect)
    list_drug_sideeffect = (drug_sideeffect.row, drug_sideeffect.col)
    list_sideeffect_drug = (drug_sideeffect.col, drug_sideeffect.row)

    drug_disease = coo_matrix(drug_disease)
    list_drug_disease = (drug_disease.row, drug_disease.col)
    list_disease_drug = (drug_disease.col, drug_disease.row)

    protein_disease = coo_matrix(protein_disease)
    list_protein_disease = (protein_disease.row, protein_disease.col)
    list_disease_protein = (protein_disease.col, protein_disease.row)

    list_protein_sideeffect = []
    list_sideeffect_protein = []

    list_disease_sideeffect = []
    list_sideeffect_disease = []

    list_drug_protein_a = []
    list_protein_drug_a = []

    g_HIN = dgl.heterograph({(disease, DI_DI_V, disease): list_disease,
                             (drug, DR_DR_V, drug): list_drug,
                             (protein, PR_PR_V, protein): list_protein,
                             (sideeffect, SE_SE_V, sideeffect): list_sideeffect,

                             (drug, DR_DR_A, drug): list_DDI,
                             (drug, DR_PR_I, protein): list_drug_protein,
                             (drug, DR_PR_A, protein): list_drug_protein_a,
                             (drug, DR_SE_A, sideeffect): list_drug_sideeffect,
                             (drug, DR_DI_A, disease): list_drug_disease,

                             (protein, PR_DR_I, drug): list_protein_drug,
                             (protein, PR_DR_A, drug): list_protein_drug_a,
                             (protein, PR_PR_A, protein): list_PPI,
                             (protein, PR_SE_A, sideeffect): list_protein_sideeffect,
                             (protein, PR_DI_A, disease): list_protein_disease,

                             (sideeffect, SE_DR_A, drug): list_sideeffect_drug,
                             (sideeffect, SE_PR_A, protein): list_sideeffect_protein,
                             (sideeffect, SE_SE_A, sideeffect): list_SESEI,
                             (sideeffect, SE_DI_A, disease): list_sideeffect_disease,

                             (disease, DI_DR_A, drug): list_disease_drug,
                             (disease, DI_PR_A, protein): list_disease_protein,
                             (disease, DI_SE_A, sideeffect): list_disease_sideeffect,
                             (disease, DI_DI_A, disease): list_DIDII,
                             })
    if CO:
        g = g_HIN.edge_type_subgraph([DR_DR_A, DR_PR_I, DR_SE_A, DR_DI_A,
                                      PR_DR_I, PR_PR_A, PR_DI_A,
                                      SE_DR_A,
                                      DI_DR_A, DI_PR_A])
    else:
        g = g_HIN.edge_type_subgraph([DR_DR_A, DR_PR_I, DR_PR_A, DR_SE_A, DR_DI_A,
                                      PR_PR_A, PR_DR_I, PR_DR_A, PR_SE_A, PR_DI_A,
                                      SE_DR_A, SE_PR_A, SE_DI_A, SE_SE_A,
                                      DI_DR_A, DI_PR_A, DI_DI_A, DI_SE_A
                                      ])
    return g


def numConvert(num: int) -> str:
    nid = num
    # 在异构转同构的过程中,num_nodes_per_ntype为[5603,708,1512,4192] 分别对应di drug pro se
    if num < disease_len:
        nid = "DI" + str(num)
    elif disease_len <= num < disease_len + drug_len:
        nid = "DR" + str(num - disease_len)
    elif disease_len + drug_len <= num < disease_len + drug_len + protein_len:
        nid = "PR" + str(num - disease_len - drug_len)
    elif disease_len + drug_len + protein_len <= num:
        nid = "SE" + str(num - disease_len - drug_len - protein_len)
    return nid


def compute_auc_aupr(pre, target):
    roc_auc = roc_auc_score(target.detach().cpu().numpy(), pre.detach().cpu().numpy())
    precision, recall, threshold = precision_recall_curve(target.detach().cpu().numpy(), pre.detach().cpu().numpy())
    aupr = auc(recall, precision)
    return roc_auc, aupr


def l2_norm(t, axit=1):
    t = t.float()
    norm = th.norm(t, 2, axit, True) + 1e-12
    output = th.div(t, norm)
    output[th.isnan(output) | th.isinf(output)] = 0.0
    return output


def row_normalize(t):
    t = t.float()
    row_sums = t.sum(1) + 1e-12
    output = t / row_sums[:, None]
    output[th.isnan(output) | th.isinf(output)] = 0.0
    return output


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_self_loop = sp.diags((1 / (1 + rowsum)).flatten())
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    new_adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt) + d_self_loop
    return new_adj.tocoo()
