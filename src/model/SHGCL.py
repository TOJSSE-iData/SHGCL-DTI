import torch as th
import torch.nn as nn

from layers.DistMult import DistMult
from layers.sc_encoder import ScEncoder
from tools.tools import l2_norm
from layers.contrast import Contrast
from layers.mp_encoder import MpEncoder

drug = 'drug'
protein = 'protein'
disease = 'disease'
sideeffect = 'sideeffect'


class SHGCL(nn.Module):
    def __init__(self, hid_dim, args, keys, mps_len_dict: dict, attn_drop, feat_dim: dict):
        super(SHGCL, self).__init__()
        self.device = th.device(args.device)
        self.dim_embedding = hid_dim
        self.keys = keys
        self.reg_lambda = args.reg_lambda

        self.fc_dict = nn.ModuleDict({k: nn.Linear(v, hid_dim) for k, v in feat_dim.items()})
        self.scencoder = ScEncoder(hid_dim, keys)
        self.scencoder2 = ScEncoder(hid_dim, keys)
        self.mpencoder = nn.ModuleDict({k: MpEncoder(v, hid_dim, attn_drop) for k, v in mps_len_dict.items()})
        self.mpencoder2 = nn.ModuleDict({k: MpEncoder(v, hid_dim, attn_drop) for k, v in mps_len_dict.items()})
        self.constrast = Contrast(hid_dim, args.tau, keys)
        self.distmult = DistMult(self.dim_embedding)
        self.reset_parameters()

    def reset_parameters(self):
        for m in SHGCL.modules(self):
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, mean=0, std=0.1)
                if m.bias is not None:
                    m.bias.data.fill_(0.1)

    def forward(self, drug_drug, drug_chemical, drug_disease, drug_sideeffect, protein_protein,
                protein_sequence, protein_disease, drug_protein, drug_protein_mask,
                mps_dict: dict, pos_dict: dict, cl, node_feature: dict):
        node_f = {k: self.fc_dict[k](node_feature[k]) for k, v in node_feature.items()}
        node_sc, node_mp = node_f, node_f
        node_sc = self.scencoder(drug_drug, drug_chemical, drug_disease, drug_sideeffect, protein_protein,
                                 protein_sequence, protein_disease, drug_protein, node_sc)
        node_sc = self.scencoder2(drug_drug, drug_chemical, drug_disease, drug_sideeffect, protein_protein,
                                  protein_sequence, protein_disease, drug_protein, node_sc)
        node_mp = {k: self.mpencoder[k](node_mp[k], mps_dict[k]) for k, v in mps_dict.items()}
        node_mp = {k: self.mpencoder2[k](node_mp[k], mps_dict[k]) for k, v in mps_dict.items()}

        node_sc, node_mp = {k: l2_norm(v) for k, v in node_sc.items()}, {k: l2_norm(v) for k, v in node_mp.items()}
        cl_loss = self.constrast(node_sc, node_mp, pos_dict)
        node_act = node_sc
        disease_vector = node_act[disease]
        drug_vector = node_act[drug]
        protein_vector = node_act[protein]
        sideeffect_vector = node_act[sideeffect]

        mloss, dti_re = self.distmult(drug_vector, disease_vector, sideeffect_vector, protein_vector,
                                      drug_drug, drug_chemical, drug_disease, drug_sideeffect, protein_protein,
                                      protein_sequence, protein_disease, drug_protein, drug_protein_mask)
        L2_loss = 0.
        for name, param in SHGCL.named_parameters(self):
            if 'bias' not in name:
                L2_loss = L2_loss + th.sum(param.pow(2))
        L2_loss = L2_loss * 0.5
        loss = mloss + self.reg_lambda * L2_loss + cl * cl_loss
        return loss, dti_re.detach()
