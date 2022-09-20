import torch as th
import torch.nn as nn

from tools.tools import args


class DistMult(nn.Module):
    def __init__(self, dim_embedding):
        super(DistMult, self).__init__()
        self.dim_embedding = dim_embedding
        tmp = th.randn(self.dim_embedding).float()
        self.re_DDI = nn.Parameter(th.diag(th.nn.init.normal_(tmp, mean=0, std=0.1)))
        self.re_D_ch = nn.Parameter(th.diag(th.nn.init.normal_(tmp, mean=0, std=0.1)))
        self.re_D_Di = nn.Parameter(th.diag(th.nn.init.normal_(tmp, mean=0, std=0.1)))
        self.re_D_Side = nn.Parameter(th.diag(th.nn.init.normal_(tmp, mean=0, std=0.1)))
        self.re_D_P = nn.Parameter(th.diag(th.nn.init.normal_(tmp, mean=0, std=0.1)))
        self.re_PPI = nn.Parameter(th.diag(th.nn.init.normal_(tmp, mean=0, std=0.1)))
        self.re_P_seq = nn.Parameter(th.diag(th.nn.init.normal_(tmp, mean=0, std=0.1)))
        self.re_P_Di = nn.Parameter(th.diag(th.nn.init.normal_(tmp, mean=0, std=0.1)))

    def forward(self, drug_vector, disease_vector, sideeffect_vector, protein_vector,
                drug_drug, drug_chemical, drug_disease, drug_sideeffect, protein_protein,
                protein_sequence, protein_disease, drug_protein, drug_protein_mask):
        drug_drug_reconstruct = th.mm(th.mm(drug_vector, self.re_DDI), drug_vector.t())
        drug_drug_reconstruct_loss = th.sum(
            (drug_drug_reconstruct - drug_drug.float()) ** 2)

        drug_chemical_reconstruct = th.mm(th.mm(drug_vector, self.re_D_ch), drug_vector.t())
        drug_chemical_reconstruct_loss = th.sum(
            (drug_chemical_reconstruct - drug_chemical.float()) ** 2)

        drug_disease_reconstruct = th.mm(th.mm(drug_vector, self.re_D_Di), disease_vector.t())
        drug_disease_reconstruct_loss = th.sum(
            (drug_disease_reconstruct - drug_disease.float()) ** 2)

        drug_sideeffect_reconstruct = th.mm(th.mm(drug_vector, self.re_D_Side), sideeffect_vector.t())
        drug_sideeffect_reconstruct_loss = th.sum(
            (drug_sideeffect_reconstruct - drug_sideeffect.float()) ** 2)

        protein_protein_reconstruct = th.mm(th.mm(protein_vector, self.re_PPI), protein_vector.t())
        protein_protein_reconstruct_loss = th.sum(
            (protein_protein_reconstruct - protein_protein.float()) ** 2)

        protein_sequence_reconstruct = th.mm(th.mm(protein_vector, self.re_P_seq), protein_vector.t())
        protein_sequence_reconstruct_loss = th.sum(
            (protein_sequence_reconstruct - protein_sequence.float()) ** 2)

        protein_disease_reconstruct = th.mm(th.mm(protein_vector, self.re_P_Di), disease_vector.t())
        protein_disease_reconstruct_loss = th.sum(
            (protein_disease_reconstruct - protein_disease.float()) ** 2)

        drug_protein_reconstruct = th.mm(th.mm(drug_vector, self.re_D_P), protein_vector.t())
        tmp = th.mul(drug_protein_mask.float(), (drug_protein_reconstruct - drug_protein.float()))
        drug_protein_reconstruct_loss = th.sum(tmp ** 2)

        edge_mask = args.edge_mask
        if edge_mask == 'drug':
            drug_drug_reconstruct_loss = 0
        elif edge_mask == 'protein':
            protein_protein_reconstruct_loss = 0
        elif edge_mask == 'drug,protein':
            drug_drug_reconstruct_loss = 0
            protein_protein_reconstruct_loss = 0
        elif edge_mask == 'disease':
            drug_disease_reconstruct_loss = 0
            protein_disease_reconstruct_loss = 0
        elif edge_mask == 'sideeffect':
            drug_sideeffect_reconstruct_loss = 0
        elif edge_mask == 'disease,sideeffect':
            drug_disease_reconstruct_loss = 0
            protein_disease_reconstruct_loss = 0
            drug_sideeffect_reconstruct_loss = 0
        elif edge_mask == 'drugsim':
            drug_chemical_reconstruct_loss = 0
        elif edge_mask == 'proteinsim':
            protein_sequence_reconstruct_loss = 0
        elif edge_mask == 'drugsim,proteinsim':
            drug_chemical_reconstruct_loss = 0
            protein_sequence_reconstruct_loss = 0

        other_loss = drug_drug_reconstruct_loss + drug_chemical_reconstruct_loss + drug_disease_reconstruct_loss + \
                     drug_sideeffect_reconstruct_loss + protein_protein_reconstruct_loss + \
                     protein_sequence_reconstruct_loss + protein_disease_reconstruct_loss
        tloss = drug_protein_reconstruct_loss + other_loss
        return tloss, drug_protein_reconstruct
