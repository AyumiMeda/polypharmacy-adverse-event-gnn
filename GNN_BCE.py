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
        # Get the edge indices for the batch
        patient_idx, pt_idx = batch.edge_label_index_dict[('patient', 'targets', 'pt')]
        patient_emb = x_dict['patient'][patient_idx]
        pt_emb = x_dict['pt'][pt_idx]
        # Concatenate all
        edge_input = torch.cat([patient_emb, pt_emb], dim=1)
        scores = self.edge_predictor(edge_input)
        return scores

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
def validate(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    for batch in loader:
        batch = batch.to(device)
        preds = model(batch).view(-1)
        labels = batch.edge_label_dict[('patient', 'targets', 'pt')].to(device)
        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    # Convert logits to probabilities
    probs = 1 / (1 + np.exp(-all_preds))

    # Metrics
    auroc = roc_auc_score(all_labels, probs)
    auprc = average_precision_score(all_labels, probs)

    # Tune threshold for best F1
    precision, recall, thresholds = precision_recall_curve(all_labels, probs)
    f1s = (2 * precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-12)
    best_idx = np.argmax(f1s)
    best_thr = thresholds[best_idx]
    best_f1 = f1s[best_idx]

    return {"AUROC": auroc, "AUPRC": auprc, "Best_F1": best_f1, "Best_thr": best_thr}

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
        add_negative_train_samples=True,
        neg_sampling_ratio= 5,
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

    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([5.0], device=device))

    for epoch in range(5):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
            batch = batch.to(device)
            optimiser.zero_grad()
            edge_logits = model(batch).view(-1)  # [num_edges]
            labels = batch.edge_label_dict[('patient', 'targets', 'pt')].float().to(device)
            loss = criterion(edge_logits, labels)  # [num_edges] vs [num_edges]
            loss.backward()
            optimiser.step()
            total_loss += loss.item()
            del batch
            torch.cuda.empty_cache()

        # Validation after each epoch
        val_metrics = validate(model, valid_loader, device)
        print(f"Epoch {epoch} Loss: {total_loss / len(train_loader):.4f}")
        print(val_metrics)

    torch.save(model.state_dict(), "gnn_model_epoch10_bce.pth")


main()
