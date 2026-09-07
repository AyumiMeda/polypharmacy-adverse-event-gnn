from pathlib import Path
import random

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import HeteroConv, SAGEConv, NNConv
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    classification_report,
)
from tqdm import tqdm


SEED = 42
HIDDEN_CHANNELS = 128
NUM_GNN_LAYERS = 3
NUM_NEIGHBORS = [10, 7, 5]
BATCH_SIZE = 256
NUM_EPOCHS = 30
PATIENCE = 5
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
DROPOUT = 0.2
POS_WEIGHT_MAX = 25.0

OUTCOME_CODES = [
    'DE',
    'LT',
    'HO',
    'DS',
    'CA',
    'RI',
    'OT',
]


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_torch_load(path):
    try:
        return torch.load(path, weights_only=False)
    except TypeError:
        return torch.load(path)


def edge_network(hidden_channels):
    return nn.Sequential(
        nn.Linear(hidden_channels, hidden_channels),
        nn.ReLU(),
        nn.Linear(hidden_channels, hidden_channels * hidden_channels),
    )


def sanitize_fixed_tensor(name, tensor):
    if not torch.is_tensor(tensor):
        return tensor

    if torch.is_floating_point(tensor):
        bad = ~torch.isfinite(tensor)
        count = int(bad.sum().item())
        if count:
            print(f'WARNING: {name} contains {count} NaN/Inf values; replacing them with 0.')
            tensor = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)

    return tensor.detach()


class advGNN(nn.Module):
    def __init__(
        self,
        hidden_channels,
        chemical_feat_dim,
        demo_edge_feat_dim,
        num_genes,
        side_effect_emb,
        num_mfr,
        num_occr,
        num_sex,
        patient_num_dim,
        num_outcomes,
        patient_num_mean,
        patient_num_std,
        num_layers=3,
        dropout=0.2,
    ):
        super().__init__()

        self.hidden_channels = hidden_channels
        self.num_outcomes = num_outcomes
        self.dropout = nn.Dropout(dropout)

        self.side_effect_emb = side_effect_emb
        self.side_effect_emb.weight.requires_grad = False
        self.side_effect_proj = nn.Sequential(
            nn.Linear(self.side_effect_emb.embedding_dim, hidden_channels),
            nn.ReLU(),
        )

        self.gene_emb = nn.Embedding(num_genes, hidden_channels)

        self.mfr_emb = nn.Embedding(num_mfr, 8)
        self.occr_emb = nn.Embedding(num_occr, 6)
        self.sex_emb = nn.Embedding(num_sex, 4)

        patient_feat_dim = 8 + 6 + 4 + patient_num_dim
        self.patient_proj = nn.Sequential(
            nn.Linear(patient_feat_dim, hidden_channels),
            nn.ReLU(),
        )

        self.chem_proj = nn.Sequential(
            nn.Linear(chemical_feat_dim, hidden_channels),
            nn.ReLU(),
        )
        self.drug_proj = nn.Sequential(
            nn.Linear(768, hidden_channels),
            nn.ReLU(),
        )
        self.indi_pt_proj = nn.Sequential(
            nn.Linear(768, hidden_channels),
            nn.ReLU(),
        )
        self.disease_proj = nn.Sequential(
            nn.Linear(768, hidden_channels),
            nn.ReLU(),
        )

        self.demo_edge_proj = nn.Sequential(
            nn.Linear(demo_edge_feat_dim, hidden_channels),
            nn.ReLU(),
        )

        self.register_buffer('patient_num_mean', patient_num_mean)
        self.register_buffer('patient_num_std', patient_num_std)

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                HeteroConv({
                    ('chemical', 'targets', 'chemical'): NNConv(
                        in_channels=hidden_channels,
                        out_channels=hidden_channels,
                        nn=edge_network(hidden_channels),
                        aggr='mean',
                    ),
                    ('chemical', 'rev_targets', 'chemical'): NNConv(
                        in_channels=hidden_channels,
                        out_channels=hidden_channels,
                        nn=edge_network(hidden_channels),
                        aggr='mean',
                    ),
                    ('chemical', 'targets', 'gene'): SAGEConv(
                        (hidden_channels, hidden_channels), hidden_channels
                    ),
                    ('gene', 'rev_targets', 'chemical'): SAGEConv(
                        (hidden_channels, hidden_channels), hidden_channels
                    ),
                    ('chemical', 'targets', 'disease_class'): SAGEConv(
                        (hidden_channels, hidden_channels), hidden_channels
                    ),
                    ('disease_class', 'rev_targets', 'chemical'): SAGEConv(
                        (hidden_channels, hidden_channels), hidden_channels
                    ),
                    ('drugs', 'targets', 'chemical'): SAGEConv(
                        (hidden_channels, hidden_channels), hidden_channels
                    ),
                    ('chemical', 'rev_targets', 'drugs'): SAGEConv(
                        (hidden_channels, hidden_channels), hidden_channels
                    ),
                    ('drugs', 'targets', 'indi_pt'): SAGEConv(
                        (hidden_channels, hidden_channels), hidden_channels
                    ),
                    ('indi_pt', 'rev_targets', 'drugs'): SAGEConv(
                        (hidden_channels, hidden_channels), hidden_channels
                    ),
                    ('drugs', 'rev_targets', 'patient'): NNConv(
                        in_channels=hidden_channels,
                        out_channels=hidden_channels,
                        nn=edge_network(hidden_channels),
                        aggr='mean',
                    ),
                    ('gene', 'targets', 'gene'): SAGEConv(
                        (hidden_channels, hidden_channels), hidden_channels
                    ),
                    ('gene', 'rev_targets', 'gene'): SAGEConv(
                        (hidden_channels, hidden_channels), hidden_channels
                    ),
                }, aggr='mean')
            )

        self.outcome_emb = nn.Embedding(num_outcomes, hidden_channels)

        self.edge_predictor = nn.Sequential(
            nn.Linear(2 * hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1),
        )

    def forward(self, batch):
        device = next(self.parameters()).device

        cat_idx = batch['patient'].cat_index.long()
        num_feat = batch['patient'].numerical.float()
        num_feat = torch.nan_to_num(num_feat, nan=0.0, posinf=0.0, neginf=0.0)
        num_feat = (num_feat - self.patient_num_mean) / self.patient_num_std

        mfr_e = self.mfr_emb(cat_idx[:, 0])
        occr_e = self.occr_emb(cat_idx[:, 1])
        sex_e = self.sex_emb(cat_idx[:, 2])

        patient_features = torch.cat([mfr_e, occr_e, sex_e, num_feat], dim=1)

        x_dict = {
            'patient': self.patient_proj(patient_features),
            'chemical': self.chem_proj(batch['chemical'].x.float()),
            'drugs': self.drug_proj(batch['drugs'].x.float()),
            'indi_pt': self.indi_pt_proj(batch['indi_pt'].x.float()),
            'disease_class': self.disease_proj(batch['disease_class'].x.float()),
            'gene': self.gene_emb(batch['gene'].x.long()).to(device),
        }

        chem_ids = batch['chemical', 'targets', 'chemical'].edge_attr.long().to(device)
        chem_edge_attr = self.side_effect_proj(self.side_effect_emb(chem_ids))

        rev_chem_ids = batch['chemical', 'rev_targets', 'chemical'].edge_attr.long().to(device)
        rev_chem_edge_attr = self.side_effect_proj(self.side_effect_emb(rev_chem_ids))

        rev_demo_edge_attr = self.demo_edge_proj(
            batch['drugs', 'rev_targets', 'patient'].edge_attr.float()
        )

        edge_attr_dict = {
            ('chemical', 'targets', 'chemical'): chem_edge_attr,
            ('chemical', 'rev_targets', 'chemical'): rev_chem_edge_attr,
            ('drugs', 'rev_targets', 'patient'): rev_demo_edge_attr,
        }

        for conv in self.convs:
            updated = conv(x_dict, batch.edge_index_dict, edge_attr_dict)
            x_dict = {
                node_type: self.dropout(F.relu(updated[node_type] + x_dict[node_type]))
                for node_type in updated
            }

        num_seed_patients = batch['patient'].batch_size
        patient_emb = x_dict['patient'][:num_seed_patients]

        outcome_emb = self.outcome_emb.weight
        batch_size = patient_emb.size(0)
        num_outcomes = outcome_emb.size(0)

        patient_expanded = patient_emb.unsqueeze(1).expand(
            batch_size, num_outcomes, self.hidden_channels
        )
        outcome_expanded = outcome_emb.unsqueeze(0).expand(
            batch_size, num_outcomes, self.hidden_channels
        )

        edge_input = torch.cat([patient_expanded, outcome_expanded], dim=-1)
        logits = self.edge_predictor(edge_input).squeeze(-1)

        return logits


def tune_thresholds(labels, probs):
    thresholds = np.full(labels.shape[1], 0.5, dtype=np.float32)

    for class_idx in range(labels.shape[1]):
        y_true = labels[:, class_idx]
        y_prob = probs[:, class_idx]

        if y_true.min() == y_true.max():
            continue

        precision, recall, candidate_thresholds = precision_recall_curve(
            y_true,
            y_prob,
        )

        if candidate_thresholds.size == 0:
            continue

        f1 = (
            2.0
            * precision[:-1]
            * recall[:-1]
            / (precision[:-1] + recall[:-1] + 1e-12)
        )

        best_idx = int(np.nanargmax(f1))
        thresholds[class_idx] = candidate_thresholds[best_idx]

    return thresholds


def multilabel_metrics(labels, probs, thresholds):
    preds = (probs >= thresholds.reshape(1, -1)).astype(np.int64)

    metrics = {
        'Macro_F1': f1_score(
            labels,
            preds,
            average='macro',
            zero_division=0,
        ),
        'Micro_F1': f1_score(
            labels,
            preds,
            average='micro',
            zero_division=0,
        ),
        'Weighted_F1': f1_score(
            labels,
            preds,
            average='weighted',
            zero_division=0,
        ),
        'Macro_Precision': precision_score(
            labels,
            preds,
            average='macro',
            zero_division=0,
        ),
        'Macro_Recall': recall_score(
            labels,
            preds,
            average='macro',
            zero_division=0,
        ),
    }

    try:
        metrics['Macro_AUROC'] = roc_auc_score(
            labels,
            probs,
            average='macro',
        )
    except ValueError:
        metrics['Macro_AUROC'] = float('nan')

    try:
        metrics['Macro_AUPRC'] = average_precision_score(
            labels,
            probs,
            average='macro',
        )
    except ValueError:
        metrics['Macro_AUPRC'] = float('nan')

    return metrics, preds


@torch.no_grad()
def collect_predictions(model, loader, device):
    model.eval()

    all_logits = []
    all_labels = []

    for batch in loader:
        batch = batch.to(device)
        logits = model(batch)

        num_seed_patients = batch['patient'].batch_size
        labels = batch['patient'].y[:num_seed_patients].float()

        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())

    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)

    probs = torch.sigmoid(logits).numpy()
    labels_np = labels.numpy().astype(np.int64)

    return labels_np, probs


def print_split_distribution(name, indices, labels):
    split_labels = labels[indices]
    counts = split_labels.sum(axis=0).astype(np.int64)
    prevalence = split_labels.mean(axis=0)

    print(f'\n{name} positive-label distribution:')
    for code, count, rate in zip(OUTCOME_CODES, counts, prevalence):
        print(f'  {code}: {count} ({rate:.4%})')


def print_per_label_metrics(labels, probs, preds):
    print('\nPer-label metrics:')
    print(
        f"{'Outcome':<8}"
        f"{'Precision':>12}"
        f"{'Recall':>12}"
        f"{'F1':>12}"
        f"{'AUROC':>12}"
        f"{'AUPRC':>12}"
        f"{'Support':>12}"
    )

    for idx, code in enumerate(OUTCOME_CODES):
        y_true = labels[:, idx]
        y_pred = preds[:, idx]
        y_prob = probs[:, idx]

        precision = precision_score(
            y_true,
            y_pred,
            zero_division=0,
        )
        recall = recall_score(
            y_true,
            y_pred,
            zero_division=0,
        )
        f1 = f1_score(
            y_true,
            y_pred,
            zero_division=0,
        )

        try:
            auroc = roc_auc_score(y_true, y_prob)
        except ValueError:
            auroc = float('nan')

        try:
            auprc = average_precision_score(y_true, y_prob)
        except ValueError:
            auprc = float('nan')

        support = int(y_true.sum())

        print(
            f'{code:<8}'
            f'{precision:>12.4f}'
            f'{recall:>12.4f}'
            f'{f1:>12.4f}'
            f'{auroc:>12.4f}'
            f'{auprc:>12.4f}'
            f'{support:>12}'
        )


def main():
    set_seed()
    cwd = Path.cwd()

    data = safe_torch_load(
        cwd / 'output_csv' / 'graph_data' / 'graph_data.pt'
    )
    emb_map = safe_torch_load(
        cwd / 'output_csv' / 'pre-embeddings' / 'pre_embeddings.pt'
    )

    if 'y' not in data['patient']:
        raise ValueError(
            "graph_data.pt does not contain data['patient'].y."
        )

    if data['patient'].y.dim() != 2 or data['patient'].y.size(1) != len(OUTCOME_CODES):
        raise ValueError(
            "Expected data['patient'].y to have shape [num_patients, 7] "
            "for [DE, LT, HO, DS, CA, RI, OT]."
        )

    data['patient'].y = sanitize_fixed_tensor(
        'patient.y',
        data['patient'].y.float(),
    )

    if not torch.all(
        (data['patient'].y == 0) | (data['patient'].y == 1)
    ):
        raise ValueError(
            "data['patient'].y must contain only binary 0/1 outcome labels."
        )

    data['patient'].num_nodes = int(data['patient'].y.size(0))

    if data['patient'].cat_index.size(1) != 3:
        raise ValueError(
            'Expected exactly 3 patient categorical columns '
            '[manufacturer, occurrence country, sex].'
        )

    data['chemical'].x = sanitize_fixed_tensor(
        'chemical.x',
        data['chemical'].x,
    )
    data['drugs'].x = sanitize_fixed_tensor(
        'drugs.x',
        data['drugs'].x,
    )
    data['indi_pt'].x = sanitize_fixed_tensor(
        'indi_pt.x',
        data['indi_pt'].x,
    )
    data['disease_class'].x = sanitize_fixed_tensor(
        'disease_class.x',
        data['disease_class'].x,
    )
    data['patient'].numerical = sanitize_fixed_tensor(
        'patient.numerical',
        data['patient'].numerical,
    )

    disease_edge = ('chemical', 'targets', 'disease_class')
    reverse_disease_edge = ('disease_class', 'rev_targets', 'chemical')
    if reverse_disease_edge not in data.edge_types:
        data[reverse_disease_edge].edge_index = (
            data[disease_edge].edge_index.flip(0)
        )

    patient_to_drug_edge = ('patient', 'targets', 'drugs')
    reverse_patient_drug_edge = ('drugs', 'rev_targets', 'patient')

    data[patient_to_drug_edge].edge_attr = sanitize_fixed_tensor(
        'patient->drugs.edge_attr',
        data[patient_to_drug_edge].edge_attr,
    )

    data[reverse_patient_drug_edge].edge_attr = (
        data[patient_to_drug_edge].edge_attr
    )
    del data[patient_to_drug_edge]

    labels_np = data['patient'].y.cpu().numpy().astype(np.int64)
    all_indices = np.arange(data['patient'].num_nodes)

    train_idx, temp_idx = train_test_split(
        all_indices,
        test_size=0.20,
        random_state=SEED,
        shuffle=True,
    )

    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.50,
        random_state=SEED,
        shuffle=True,
    )

    print_split_distribution(
        'Train',
        train_idx,
        labels_np,
    )
    print_split_distribution(
        'Validation',
        val_idx,
        labels_np,
    )
    print_split_distribution(
        'Test',
        test_idx,
        labels_np,
    )

    train_idx_t = torch.tensor(train_idx, dtype=torch.long)
    val_idx_t = torch.tensor(val_idx, dtype=torch.long)
    test_idx_t = torch.tensor(test_idx, dtype=torch.long)

    train_num = data['patient'].numerical[train_idx_t].float()
    train_num = torch.nan_to_num(
        train_num,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    patient_num_mean = train_num.mean(dim=0)
    patient_num_std = train_num.std(dim=0).clamp_min(1e-6)

    if len(NUM_NEIGHBORS) != NUM_GNN_LAYERS:
        raise ValueError(
            'NUM_NEIGHBORS must have one entry per GNN layer.'
        )

    neighbor_spec = {
        edge_type: NUM_NEIGHBORS
        for edge_type in data.edge_types
    }

    train_loader = NeighborLoader(
        data,
        num_neighbors=neighbor_spec,
        input_nodes=('patient', train_idx_t),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    valid_loader = NeighborLoader(
        data,
        num_neighbors=neighbor_spec,
        input_nodes=('patient', val_idx_t),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    test_loader = NeighborLoader(
        data,
        num_neighbors=neighbor_spec,
        input_nodes=('patient', test_idx_t),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    side_effect_tensor = torch.stack(
        list(emb_map['side_effect'].values())
    )
    side_effect_emb = nn.Embedding.from_pretrained(
        side_effect_tensor,
        freeze=True,
    )

    patient_cat = data['patient'].cat_index
    num_mfr = int(patient_cat[:, 0].max().item()) + 1
    num_occr = int(patient_cat[:, 1].max().item()) + 1
    num_sex = int(patient_cat[:, 2].max().item()) + 1

    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )
    print(f'\nUsing device: {device}')

    model = advGNN(
        hidden_channels=HIDDEN_CHANNELS,
        chemical_feat_dim=data['chemical'].x.size(1),
        demo_edge_feat_dim=(
            data['drugs', 'rev_targets', 'patient']
            .edge_attr
            .size(1)
        ),
        num_genes=data['gene'].num_nodes,
        side_effect_emb=side_effect_emb,
        num_mfr=num_mfr,
        num_occr=num_occr,
        num_sex=num_sex,
        patient_num_dim=data['patient'].numerical.size(1),
        num_outcomes=len(OUTCOME_CODES),
        patient_num_mean=patient_num_mean,
        patient_num_std=patient_num_std,
        num_layers=NUM_GNN_LAYERS,
        dropout=DROPOUT,
    ).to(device)

    train_targets = data['patient'].y[train_idx_t].float()
    positive_counts = train_targets.sum(dim=0)
    negative_counts = train_targets.size(0) - positive_counts

    raw_pos_weight = (
        negative_counts
        / positive_counts.clamp_min(1.0)
    )

    pos_weight = torch.sqrt(raw_pos_weight)
    pos_weight = pos_weight.clamp(
        min=1.0,
        max=POS_WEIGHT_MAX,
    )

    print('\nBCE positive-class weights:')
    for code, positives, weight in zip(
        OUTCOME_CODES,
        positive_counts,
        pos_weight,
    ):
        print(
            f'  {code}: '
            f'positives={int(positives.item())} | '
            f'pos_weight={weight.item():.4f}'
        )

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight.to(device)
    )

    optimiser = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    checkpoint_path = cwd / 'gnn_multilabel_outcome_best.pth'
    best_val_auprc = -1.0
    epochs_without_improvement = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for batch in tqdm(
            train_loader,
            desc=f'Epoch {epoch}/{NUM_EPOCHS}',
        ):
            batch = batch.to(device)
            optimiser.zero_grad()

            logits = model(batch)

            if not torch.isfinite(logits).all():
                raise RuntimeError(
                    'Non-finite logits detected during training.'
                )

            num_seed_patients = batch['patient'].batch_size
            labels = (
                batch['patient']
                .y[:num_seed_patients]
                .float()
            )

            loss = criterion(logits, labels)

            if not torch.isfinite(loss):
                raise RuntimeError(
                    'Non-finite loss detected during training.'
                )

            loss.backward()
            optimiser.step()

            total_loss += loss.item()

        avg_loss = total_loss / max(len(train_loader), 1)

        val_labels, val_probs = collect_predictions(
            model,
            valid_loader,
            device,
        )

        fixed_thresholds = np.full(
            len(OUTCOME_CODES),
            0.5,
            dtype=np.float32,
        )

        val_metrics, _ = multilabel_metrics(
            val_labels,
            val_probs,
            fixed_thresholds,
        )

        print(f'\nEpoch {epoch} Loss: {avg_loss:.4f}')
        print(
            'Validation @ threshold 0.5: '
            f"Macro F1={val_metrics['Macro_F1']:.4f} | "
            f"Micro F1={val_metrics['Micro_F1']:.4f} | "
            f"Macro Precision={val_metrics['Macro_Precision']:.4f} | "
            f"Macro Recall={val_metrics['Macro_Recall']:.4f} | "
            f"Macro AUROC={val_metrics['Macro_AUROC']:.4f} | "
            f"Macro AUPRC={val_metrics['Macro_AUPRC']:.4f}"
        )

        val_auprc = val_metrics['Macro_AUPRC']

        if np.isfinite(val_auprc) and val_auprc > best_val_auprc:
            best_val_auprc = val_auprc
            epochs_without_improvement = 0

            torch.save({
                'model_state_dict': model.state_dict(),
                'outcome_codes': OUTCOME_CODES,
                'hidden_channels': HIDDEN_CHANNELS,
                'num_gnn_layers': NUM_GNN_LAYERS,
                'best_validation_macro_auprc': best_val_auprc,
            }, checkpoint_path)

            print(
                f'Saved new best model to '
                f'{checkpoint_path.name}'
            )
        else:
            epochs_without_improvement += 1

            if epochs_without_improvement >= PATIENCE:
                print(
                    f'Early stopping: validation Macro AUPRC '
                    f'did not improve for {PATIENCE} epochs.'
                )
                break

    checkpoint = safe_torch_load(checkpoint_path)
    model.load_state_dict(
        checkpoint['model_state_dict']
    )

    val_labels, val_probs = collect_predictions(
        model,
        valid_loader,
        device,
    )

    tuned_thresholds = tune_thresholds(
        val_labels,
        val_probs,
    )

    print('\nValidation-tuned thresholds:')
    for code, threshold in zip(
        OUTCOME_CODES,
        tuned_thresholds,
    ):
        print(
            f'  {code}: {threshold:.4f}'
        )

    test_labels, test_probs = collect_predictions(
        model,
        test_loader,
        device,
    )

    test_metrics, test_preds = multilabel_metrics(
        test_labels,
        test_probs,
        tuned_thresholds,
    )

    print(
        '\nBest validation Macro AUPRC:',
        f'{best_val_auprc:.4f}',
    )

    print('\nTest metrics using validation-tuned thresholds:')
    for key, value in test_metrics.items():
        print(
            f'  {key}: {value:.4f}'
        )

    print_per_label_metrics(
        test_labels,
        test_probs,
        test_preds,
    )

    print('\nTest multilabel classification report:')
    print(
        classification_report(
            test_labels,
            test_preds,
            target_names=OUTCOME_CODES,
            digits=4,
            zero_division=0,
        )
    )

    checkpoint['decision_thresholds'] = {
        code: float(threshold)
        for code, threshold in zip(
            OUTCOME_CODES,
            tuned_thresholds,
        )
    }

    torch.save(
        checkpoint,
        checkpoint_path,
    )

    print(
        f'\nBest model saved to: '
        f'{checkpoint_path}'
    )


if __name__ == '__main__':
    main()
