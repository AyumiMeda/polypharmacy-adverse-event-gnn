import torch
from torch import nn
import numpy as np
from torch_geometric.data import HeteroData
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, precision_recall_curve
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.nn import HeteroConv, SAGEConv, NNConv
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from torch_geometric import transforms as T
import torch.nn.functional as F
from torchmetrics.classification import AUROC, AveragePrecision, F1Score
from torch_geometric.transforms import RandomLinkSplit

class advGNN(nn.Module):
    def __init__(self, hidden_channels,
                 patient_feat_dim, chemical_feat_dim,
                 demo_edge_feat_dim, num_genes,
                 side_effect_emb,
                 rept_emb, mfr_emb, occr_emb, sex_emb):
        super().__init__()

        # Define embeddings
        self.gene_emb = nn.Embedding(num_genes, 128)
        self.side_effect_emb = side_effect_emb
        # Freeze embeddings
        self.side_effect_emb.weight.requires_grad = False
        self.side_effect_proj = nn.Linear(self.side_effect_emb.embedding_dim, 128)
        side_effect_dim = 128

        # Chem projections TEMPORARY
        self.chem_proj = nn.Linear(chemical_feat_dim, 128)

        # Patient projections
        self.patient_proj = nn.Linear(patient_feat_dim, 128)
        self.demo_edge_proj = nn.Linear(demo_edge_feat_dim, 128)

        # Drug projections
        self.drug_proj = nn.Linear(768, 128)

        # Patient embeddings
        self.rept_emb = rept_emb
        self.mfr_emb = mfr_emb
        self.occr_emb = occr_emb
        self.sex_emb = sex_emb


        # Edge networks for NNConv
        chem_edge_network = nn.Sequential(
            nn.Linear(side_effect_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128 * hidden_channels)
        )
        demo_edge_network = nn.Sequential(
            nn.Linear(128, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 128 * 128)
        )
        # HeteroConv layer
        self.conv1 = HeteroConv({
            ('chemical', 'targets', 'chemical'): NNConv(
                in_channels=128,
                out_channels=hidden_channels,
                nn=chem_edge_network,
                aggr='mean'
            ),
            ('chemical', 'rev_targets', 'chemical'): NNConv(
                in_channels=128,
                out_channels=hidden_channels,
                nn=chem_edge_network,
                aggr='mean'
            ),
            ('chemical', 'targets', 'gene'): SAGEConv((-1, -1), hidden_channels),
            ('gene', 'rev_targets', 'chemical'): SAGEConv((-1, -1), hidden_channels),
            ('chemical', 'targets', 'disease_class'): SAGEConv((-1, -1), hidden_channels),
            ('drugs', 'targets', 'chemical'): SAGEConv((-1, -1), hidden_channels),
            ('chemical', 'rev_targets', 'drugs'): SAGEConv((-1, -1), hidden_channels),
            ('drugs', 'targets', 'indi_pt'): SAGEConv((-1, -1), hidden_channels),
            ('indi_pt', 'rev_targets', 'drugs'): SAGEConv((-1, -1), hidden_channels),
            ('patient', 'targets', 'drugs'): NNConv(
                in_channels=128,
                out_channels=hidden_channels,
                nn=demo_edge_network,
                aggr='mean'
            ),
            ('patient', 'targets', 'pt'): SAGEConv((-1, -1), hidden_channels),
            ('drugs', 'rev_targets', 'patient'): SAGEConv((-1, -1), hidden_channels),
            ('gene', 'targets', 'gene'): SAGEConv((-1, -1), hidden_channels),
            ('gene', 'rev_targets', 'gene'): SAGEConv((-1, -1), hidden_channels),
            ('pt', 'targets', 'disease_class'): SAGEConv((-1, -1), hidden_channels),
        }, aggr='mean')

        self.edge_predictor = nn.Sequential(
            nn.Linear(2 * hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1)
        )

    def forward(self, batch):
        device = next(self.parameters()).device

        # Copy x_dict to acoid in memory modification
        x_dict = batch.x_dict.copy()
        edge_index_dict = batch.edge_index_dict
        edge_attr_dict = batch.edge_attr_dict

        # Embed patient node features
        cat_idx = x_dict['patient'][:, :4].long()
        num_feat = x_dict['patient'][:, 4:]
        rept_e = self.rept_emb(cat_idx[:, 0])
        mfr_e = self.mfr_emb(cat_idx[:, 1])
        occr_e = self.occr_emb(cat_idx[:, 2])
        sex_e = self.sex_emb(cat_idx[:, 3])

        # Concatenate embeddings + numerical features
        x_dict['patient'] = torch.cat([rept_e, mfr_e, occr_e, sex_e, num_feat], dim=1)

        # Embed gene nodes and side effect edges
        x_dict['gene'] = self.gene_emb(x_dict['gene'].long()).to(device)
        ids = edge_attr_dict[('chemical', 'targets', 'chemical')].to(device)
        edge_emb = self.side_effect_emb(ids)
        edge_attr_dict[('chemical', 'targets', 'chemical')] = self.side_effect_proj(edge_emb)
        ids = edge_attr_dict[('chemical', 'rev_targets', 'chemical')].to(device)
        rev_edge_emb = self.side_effect_emb(ids)
        edge_attr_dict[('chemical', 'rev_targets', 'chemical')] = self.side_effect_proj(rev_edge_emb)

        # Project patient and chem embeddings to the right dimension
        x_dict['patient'] = self.patient_proj(x_dict['patient'])
        x_dict['chemical'] = self.chem_proj(x_dict['chemical'])
        x_dict['drugs'] = self.drug_proj(x_dict['drugs'])

        # Project patients edge to correct dimension
        edge_attr_dict[('patient', 'targets', 'drugs')] = self.demo_edge_proj(
            edge_attr_dict[('patient', 'targets', 'drugs')]
        )

        # Message passing
        x_dict = self.conv1(x_dict, edge_index_dict, edge_attr_dict)

        return x_dict

# ---------------- Helper: Rescale edge predictor logits ----------------
def rescale_logits(edge_logits, reference_logits):
    """
    Rescales edge predictor logits so that their mean and std
    match the distribution of the reference logits (in-batch dot products).
    """
    if edge_logits.numel() == 0 or reference_logits.numel() == 0:
        return edge_logits  # nothing to rescale

    # Compute statistics
    ref_mean, ref_std = reference_logits.mean(), reference_logits.std()
    edge_mean, edge_std = edge_logits.mean(), edge_logits.std()

    # Avoid divide-by-zero
    if edge_std < 1e-6:
        edge_std = 1.0

    # Standardize then rescale
    rescaled = (edge_logits - edge_mean) / edge_std
    rescaled = rescaled * ref_std + ref_mean
    return rescaled

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
@torch.no_grad()
def validate_full_mlp(model, loader, device, pt_bank=None, sample_bank_size=100):
    """
    Validation for sparse link prediction using MLP edge predictor + softmax loss:
    - Per-patient top-k metrics (precision, recall, F1)
    - Global AUROC and AUPRC using MLP logits
    """
    model.eval()

    all_logits_list, all_labels_list = [], []
    precisions, recalls, f1s = [], [], []

    for batch in tqdm(loader, desc="Validation"):
        batch = batch.to(device)
        x_dict = model(batch)

        # ---------------- Positive edges ----------------
        patient_idx, pt_idx = batch.edge_label_index_dict[('patient', 'targets', 'pt')]
        labels = batch.edge_label_dict[('patient', 'targets', 'pt')]

        unique_patient = torch.unique(patient_idx)
        unique_pt = torch.unique(pt_idx)

        pt2col = {pt.item(): i for i, pt in enumerate(unique_pt)}
        labels_matrix = torch.zeros((len(unique_patient), len(unique_pt)), device=device)
        for j, p in enumerate(unique_patient):
            pos_pt_nodes = pt_idx[(patient_idx == p) & (labels == 1)]
            col_indices = [pt2col[pt.item()] for pt in pos_pt_nodes]
            labels_matrix[j, col_indices] = 1
        pos_mask = labels_matrix.bool()

        # ---------------- Normalize embeddings ----------------
        patient_emb = F.normalize(x_dict['patient'][unique_patient], p=2, dim=-1)
        pt_emb = F.normalize(x_dict['pt'][unique_pt], p=2, dim=-1)

        # ---------------- Compute logits via MLP ----------------
        logits_matrix = torch.zeros_like(labels_matrix, dtype=torch.float, device=device)
        for p in range(len(unique_patient)):
            pos_idx = pos_mask[p].nonzero(as_tuple=False).flatten()
            if len(pos_idx) == 0:
                continue
            patient_rep = patient_emb[p].unsqueeze(0).repeat(len(pos_idx), 1)
            pos_pairs = torch.cat([patient_rep, pt_emb[pos_idx]], dim=1)
            logits_matrix[p, pos_idx] = model.edge_predictor(pos_pairs).flatten()

        # ---------------- Memory bank negatives ----------------
        if pt_bank is not None and pt_bank.numel() > 0:
            bank_subset = pt_bank if pt_bank.size(0) <= sample_bank_size else pt_bank[
                torch.randperm(pt_bank.size(0), device=device)[:sample_bank_size]
            ]
            bank_subset = F.normalize(bank_subset, p=2, dim=-1)

            for p in range(len(unique_patient)):
                patient_rep = patient_emb[p].unsqueeze(0).repeat(bank_subset.size(0), 1)
                bank_pairs = torch.cat([patient_rep, bank_subset], dim=1)
                bank_logits = model.edge_predictor(bank_pairs).flatten()
                logits_matrix[p, :bank_logits.size(0)] = torch.cat([logits_matrix[p, :len(bank_logits)], bank_logits])

        # ---------------- Collect global logits ----------------
        pos_logits = logits_matrix[pos_mask]
        pos_labels = torch.ones_like(pos_logits)

        if pt_bank is not None and pt_bank.numel() > 0:
            neg_logits = bank_logits.flatten()
            neg_labels = torch.zeros_like(neg_logits)
        else:
            neg_logits = torch.tensor([], device=device)
            neg_labels = torch.tensor([], device=device)

        all_logits_list.append(torch.cat([pos_logits, neg_logits]).cpu())
        all_labels_list.append(torch.cat([pos_labels, neg_labels]).cpu())

        # ---------------- Per-patient top-k ----------------
        for p_idx in range(len(unique_patient)):
            true_pos_count = pos_mask[p_idx].sum().item()
            if true_pos_count == 0:
                continue
            # Top-k using MLP logits
            top_pred_idx = logits_matrix[p_idx].topk(true_pos_count).indices
            true_idx = torch.where(pos_mask[p_idx])[0]
            correct = len(set(top_pred_idx.cpu().numpy()) & set(true_idx.cpu().numpy()))
            precision = correct / true_pos_count
            recall = correct / true_pos_count
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)

        del batch
        torch.cuda.empty_cache()

    all_probs = torch.sigmoid(torch.cat(all_logits_list)).numpy()
    all_labels = torch.cat(all_labels_list).numpy()

    auroc_val = roc_auc_score(all_labels, all_probs)
    auprc_val = average_precision_score(all_labels, all_probs)

    avg_precision = np.mean(precisions) if len(precisions) > 0 else 0.0
    avg_recall = np.mean(recalls) if len(recalls) > 0 else 0.0
    avg_f1 = np.mean(f1s) if len(f1s) > 0 else 0.0

    return {
        "AUROC": auroc_val,
        "AUPRC": auprc_val,
        "Precision_topk": avg_precision,
        "Recall_topk": avg_recall,
        "F1_topk": avg_f1
    }

# ---------------- Main ----------------
def main():
    cwd = Path.cwd()
    data = torch.load(cwd / 'output_csv' / 'graph_data' / 'graph_data.pt')
    demo = pd.read_csv(cwd / 'output_csv' / 'demo_clean' / 'demo_clean.csv')
    emb_map = torch.load(Path(cwd, 'output_csv', 'pre-embeddings', 'pre_embeddings.pt'))

    # side effect embeddings path
    side_effect_tensor = torch.stack(list(emb_map['side_effect'].values()))
    side_effect_map = {s: i for i,s in enumerate(emb_map['side_effect'].keys())}
    side_effect_emb = nn.Embedding.from_pretrained(side_effect_tensor, freeze=False)

    # patient categorical embeddings
    demo['rept_cod'] = demo['rept_cod'].fillna('UNKNOWN')
    rept_emb = nn.Embedding(len(demo['rept_cod'].unique()), 4)
    valid_sex = {'F', 'M', 'UNK'}
    demo['sex'] = demo['sex'].fillna("UNK")
    demo['sex'] = demo['sex'].apply(lambda x: x if x in valid_sex else 'UNK')
    sex_emb = nn.Embedding(3, 4)
    demo['mfr_sndr'] = demo['mfr_sndr'].fillna('UNKNOWN')
    mfr_sndr_emb = nn.Embedding(len(demo['mfr_sndr'].unique()), 8)
    demo['occr_country'] = demo['occr_country'].fillna('UNKNOWN')
    occr_emb = nn.Embedding(len(demo['occr_country'].unique()), 6)

    # Concatenate for x_dict
    data['patient'].x = torch.cat([
        torch.tensor(data['patient'].cat_index, dtype=torch.long),
        torch.tensor(data['patient'].numerical , dtype=torch.float)
    ], dim=1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = advGNN(
        hidden_channels=128,
        patient_feat_dim=24,
        chemical_feat_dim=data['chemical'].x.size(1),
        demo_edge_feat_dim=data['patient', 'targets', 'drugs'].edge_attr.size(1),
        num_genes=data['gene'].num_nodes,
        side_effect_emb=side_effect_emb,
        rept_emb=rept_emb,
        mfr_emb=mfr_sndr_emb,
        occr_emb=occr_emb,
        sex_emb=sex_emb,
    ).to(device)

    # Create training, validation, test data 80/10/10

    transform = T.RandomLinkSplit(
        num_val = 0.1,
        num_test = 0.1,
        is_undirected = True,
        add_negative_train_samples=False,
        edge_types=("patient", "targets", "pt"),
        rev_edge_types = None
    )

    train_data, val_data, test_data = transform(data)

    # LinkNeighbourLoaders
    train_loader = LinkNeighborLoader(
        train_data,
        num_neighbors={key:[10,7,5,3] for key in train_data.edge_types},
        edge_label_index=(('patient', 'targets', 'pt'),
                          train_data["patient", "targets", "pt"].edge_label_index),
        edge_label=train_data["patient", "targets", "pt"].edge_label,
        batch_size=400,
        shuffle=True,
    )
    valid_loader = LinkNeighborLoader(
        val_data,
        num_neighbors={key:[5,3,2,1] for key in val_data.edge_types},
        edge_label_index=(('patient', 'targets', 'pt'),
                          val_data["patient", "targets", "pt"].edge_label_index),
        edge_label=val_data["patient", "targets", "pt"].edge_label,
        batch_size=200,
    )
    test_loader = LinkNeighborLoader(
        data,
        num_neighbors={key: [10, 7, 5, 3] for key in test_data.edge_types},
        edge_label_index=(('patient', 'targets', 'pt'),
                          test_data["patient", "targets", "pt"].edge_label_index),
        edge_label=test_data["patient", "targets", "pt"].edge_label,
        batch_size=200,
        shuffle=True,
    )

    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.load_state_dict(torch.load('gnn_model_epoch10_bce.pth'))

    # ---------------- Hyperparameters ----------------
    bank_size = 20000
    bank_ptr = 0
    bank_filled = False
    topk_neg_per_patient = 20  # store top 100 hard negatives per patient
    sample_bank_size = 50  # use 100 negatives from the bank per batch

    pt_bank = torch.zeros((bank_size, 128), device=device)
    score_bank = torch.zeros(bank_size, device=device)

    # ---------------- Training Loop ----------------
    for epoch in range(10):
        model.train()
        total_loss = 0

        for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
            batch = batch.to(device)
            optimiser.zero_grad()

            # Forward pass to get node embeddings
            x_dict = model(batch)

            # ---------------- Positive edges ----------------
            patient_idx, pt_idx = batch.edge_label_index_dict[('patient', 'targets', 'pt')]
            labels = batch.edge_label_dict[('patient', 'targets', 'pt')]

            unique_patient = torch.unique(patient_idx)
            unique_pt = torch.unique(pt_idx)

            pt2col = {pt.item(): i for i, pt in enumerate(unique_pt)}
            labels_matrix = torch.zeros((len(unique_patient), len(unique_pt)), device=device)
            for j, p in enumerate(unique_patient):
                pos_pt_nodes = pt_idx[(patient_idx == p) & (labels == 1)]
                col_indices = [pt2col[pt.item()] for pt in pos_pt_nodes]
                labels_matrix[j, col_indices] = 1
            pos_mask = labels_matrix.bool()

            # ---------------- Normalize embeddings ----------------
            patient_emb = F.normalize(x_dict['patient'][unique_patient], p=2, dim=-1)
            pt_emb = F.normalize(x_dict['pt'][unique_pt], p=2, dim=-1)

            # ---------------- Compute in-batch logits ----------------
            logits_matrix = torch.zeros_like(labels_matrix, dtype=torch.float, device=device)
            for p in range(len(unique_patient)):
                # Positives in batch
                pos_idx = pos_mask[p].nonzero(as_tuple=False).flatten()
                if len(pos_idx) == 0:
                    continue
                patient_rep = patient_emb[p].unsqueeze(0).repeat(len(pos_idx), 1)
                pos_pairs = torch.cat([patient_rep, pt_emb[pos_idx]], dim=1)
                logits_matrix[p, pos_idx] = model.edge_predictor(pos_pairs).flatten()

            # ---------------- Memory bank negatives ----------------
            if bank_filled or bank_ptr > 0:
                bank_subset = pt_bank[:bank_ptr] if not bank_filled else pt_bank
                sample_idx = torch.randperm(bank_subset.size(0), device=device)[:sample_bank_size]
                bank_sample = F.normalize(bank_subset[sample_idx], p=2, dim=-1)

                for p in range(len(unique_patient)):
                    patient_rep = patient_emb[p].unsqueeze(0).repeat(bank_sample.size(0), 1)
                    bank_pairs = torch.cat([patient_rep, bank_sample], dim=1)
                    bank_logits = model.edge_predictor(bank_pairs).flatten()
                    logits_matrix[p, :bank_logits.size(0)] = torch.cat(
                        [logits_matrix[p, :len(bank_logits)], bank_logits])

            # ---------------- Softmax over positives + negatives ----------------
            loss = 0
            for p in range(len(unique_patient)):
                pos_idx = pos_mask[p].nonzero(as_tuple=False).flatten()
                if len(pos_idx) == 0:
                    continue
                # Extract logits for positives + sampled negatives
                logits_p = logits_matrix[p]
                labels_p = pos_mask[p].float()
                # Compute softmax loss: -log( exp(pos) / sum(exp(pos+neg)) )
                logits_selected = logits_p[pos_idx]
                all_logits = logits_p
                log_probs = F.log_softmax(all_logits, dim=0)
                loss -= log_probs[pos_idx].sum() / len(pos_idx)

            # Backprop
            loss.backward()
            optimiser.step()
            total_loss += loss.item()

            # ---------------- Update memory bank ----------------
            with torch.no_grad():
                for p_idx in range(len(unique_patient)):
                    patient_neg_mask = ~pos_mask[p_idx]
                    if patient_neg_mask.sum() == 0:
                        continue
                    patient_neg_logits = logits_matrix[p_idx, patient_neg_mask]
                    patient_neg_pt_idx = torch.arange(len(unique_pt), device=device)[patient_neg_mask]

                    k = min(topk_neg_per_patient, patient_neg_logits.numel())
                    topk_vals, topk_idx = torch.topk(patient_neg_logits, k)
                    new_bank_emb = pt_emb[patient_neg_pt_idx[topk_idx]]
                    new_bank_scores = topk_vals

                    n_new = new_bank_emb.size(0)
                    if bank_ptr + n_new > bank_size:
                        overflow = (bank_ptr + n_new) - bank_size
                        pt_bank[bank_ptr:] = new_bank_emb[:n_new - overflow].detach()
                        score_bank[bank_ptr:] = new_bank_scores[:n_new - overflow].detach()
                        pt_bank[:overflow] = new_bank_emb[n_new - overflow:].detach()
                        score_bank[:overflow] = new_bank_scores[n_new - overflow:].detach()
                        bank_ptr = overflow
                        bank_filled = True
                    else:
                        pt_bank[bank_ptr:bank_ptr + n_new] = new_bank_emb.detach()
                        score_bank[bank_ptr:bank_ptr + n_new] = new_bank_scores.detach()
                        bank_ptr += n_new

            del batch
            torch.cuda.empty_cache()

        print(f"Epoch {epoch} avg loss: {total_loss / len(train_loader):.4f}")
        val_metrics = validate_full_mlp(model, valid_loader, device, pt_bank)
        print(f"Validation metrics: {val_metrics}")

    torch.save(model.state_dict(), "gnn_model_epoch10_bank.pth")


main()
