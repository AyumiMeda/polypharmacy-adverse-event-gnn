from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import (
    HeteroConv,
    SAGEConv,
    NNConv,
)


OUTCOME_CODES = [
    'DE',
    'LT',
    'HO',
    'DS',
    'CA',
    'RI',
    'OT',
]

DROPOUT = 0.2


def safe_torch_load(path):
    try:
        return torch.load(
            path,
            weights_only=False,
        )
    except TypeError:
        return torch.load(path)


def edge_network(hidden_channels):
    return nn.Sequential(
        nn.Linear(
            hidden_channels,
            hidden_channels,
        ),
        nn.ReLU(),
        nn.Linear(
            hidden_channels,
            hidden_channels
            * hidden_channels,
        ),
    )


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
            nn.Linear(
                self.side_effect_emb.embedding_dim,
                hidden_channels,
            ),
            nn.ReLU(),
        )

        self.gene_emb = nn.Embedding(
            num_genes,
            hidden_channels,
        )

        self.mfr_emb = nn.Embedding(
            num_mfr,
            8,
        )
        self.occr_emb = nn.Embedding(
            num_occr,
            6,
        )
        self.sex_emb = nn.Embedding(
            num_sex,
            4,
        )

        patient_feat_dim = (
            8
            + 6
            + 4
            + patient_num_dim
        )

        self.patient_proj = nn.Sequential(
            nn.Linear(
                patient_feat_dim,
                hidden_channels,
            ),
            nn.ReLU(),
        )

        self.chem_proj = nn.Sequential(
            nn.Linear(
                chemical_feat_dim,
                hidden_channels,
            ),
            nn.ReLU(),
        )

        self.drug_proj = nn.Sequential(
            nn.Linear(
                768,
                hidden_channels,
            ),
            nn.ReLU(),
        )

        self.indi_pt_proj = nn.Sequential(
            nn.Linear(
                768,
                hidden_channels,
            ),
            nn.ReLU(),
        )

        self.disease_proj = nn.Sequential(
            nn.Linear(
                768,
                hidden_channels,
            ),
            nn.ReLU(),
        )

        self.demo_edge_proj = nn.Sequential(
            nn.Linear(
                demo_edge_feat_dim,
                hidden_channels,
            ),
            nn.ReLU(),
        )

        self.register_buffer(
            'patient_num_mean',
            patient_num_mean,
        )
        self.register_buffer(
            'patient_num_std',
            patient_num_std,
        )

        self.convs = nn.ModuleList()

        for _ in range(num_layers):
            self.convs.append(
                HeteroConv({
                    (
                        'chemical',
                        'targets',
                        'chemical',
                    ): NNConv(
                        in_channels=hidden_channels,
                        out_channels=hidden_channels,
                        nn=edge_network(
                            hidden_channels
                        ),
                        aggr='mean',
                    ),
                    (
                        'chemical',
                        'rev_targets',
                        'chemical',
                    ): NNConv(
                        in_channels=hidden_channels,
                        out_channels=hidden_channels,
                        nn=edge_network(
                            hidden_channels
                        ),
                        aggr='mean',
                    ),
                    (
                        'chemical',
                        'targets',
                        'gene',
                    ): SAGEConv(
                        (
                            hidden_channels,
                            hidden_channels,
                        ),
                        hidden_channels,
                    ),
                    (
                        'gene',
                        'rev_targets',
                        'chemical',
                    ): SAGEConv(
                        (
                            hidden_channels,
                            hidden_channels,
                        ),
                        hidden_channels,
                    ),
                    (
                        'chemical',
                        'targets',
                        'disease_class',
                    ): SAGEConv(
                        (
                            hidden_channels,
                            hidden_channels,
                        ),
                        hidden_channels,
                    ),
                    (
                        'disease_class',
                        'rev_targets',
                        'chemical',
                    ): SAGEConv(
                        (
                            hidden_channels,
                            hidden_channels,
                        ),
                        hidden_channels,
                    ),
                    (
                        'drugs',
                        'targets',
                        'chemical',
                    ): SAGEConv(
                        (
                            hidden_channels,
                            hidden_channels,
                        ),
                        hidden_channels,
                    ),
                    (
                        'chemical',
                        'rev_targets',
                        'drugs',
                    ): SAGEConv(
                        (
                            hidden_channels,
                            hidden_channels,
                        ),
                        hidden_channels,
                    ),
                    (
                        'drugs',
                        'targets',
                        'indi_pt',
                    ): SAGEConv(
                        (
                            hidden_channels,
                            hidden_channels,
                        ),
                        hidden_channels,
                    ),
                    (
                        'indi_pt',
                        'rev_targets',
                        'drugs',
                    ): SAGEConv(
                        (
                            hidden_channels,
                            hidden_channels,
                        ),
                        hidden_channels,
                    ),
                    (
                        'drugs',
                        'rev_targets',
                        'patient',
                    ): NNConv(
                        in_channels=hidden_channels,
                        out_channels=hidden_channels,
                        nn=edge_network(
                            hidden_channels
                        ),
                        aggr='mean',
                    ),
                    (
                        'gene',
                        'targets',
                        'gene',
                    ): SAGEConv(
                        (
                            hidden_channels,
                            hidden_channels,
                        ),
                        hidden_channels,
                    ),
                    (
                        'gene',
                        'rev_targets',
                        'gene',
                    ): SAGEConv(
                        (
                            hidden_channels,
                            hidden_channels,
                        ),
                        hidden_channels,
                    ),
                }, aggr='mean')
            )

        self.outcome_emb = nn.Embedding(
            num_outcomes,
            hidden_channels,
        )

        self.edge_predictor = nn.Sequential(
            nn.Linear(
                2 * hidden_channels,
                hidden_channels,
            ),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(
                hidden_channels,
                1,
            ),
        )

    def forward(self, batch):
        cat_idx = (
            batch['patient']
            .cat_index
            .long()
        )

        num_feat = (
            batch['patient']
            .numerical
            .float()
        )

        num_feat = torch.nan_to_num(
            num_feat,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        num_feat = (
            num_feat
            - self.patient_num_mean
        ) / self.patient_num_std

        patient_features = torch.cat([
            self.mfr_emb(
                cat_idx[:, 0]
            ),
            self.occr_emb(
                cat_idx[:, 1]
            ),
            self.sex_emb(
                cat_idx[:, 2]
            ),
            num_feat,
        ], dim=1)

        x_dict = {
            'patient': self.patient_proj(
                patient_features
            ),
            'chemical': self.chem_proj(
                batch['chemical'].x.float()
            ),
            'drugs': self.drug_proj(
                batch['drugs'].x.float()
            ),
            'indi_pt': self.indi_pt_proj(
                batch['indi_pt'].x.float()
            ),
            'disease_class': self.disease_proj(
                batch['disease_class'].x.float()
            ),
            'gene': self.gene_emb(
                batch['gene'].x.long()
            ),
        }

        chem_ids = batch[
            'chemical',
            'targets',
            'chemical',
        ].edge_attr.long()

        rev_chem_ids = batch[
            'chemical',
            'rev_targets',
            'chemical',
        ].edge_attr.long()

        rev_demo_edge_attr = self.demo_edge_proj(
            batch[
                'drugs',
                'rev_targets',
                'patient',
            ].edge_attr.float()
        )

        edge_attr_dict = {
            (
                'chemical',
                'targets',
                'chemical',
            ): self.side_effect_proj(
                self.side_effect_emb(
                    chem_ids
                )
            ),
            (
                'chemical',
                'rev_targets',
                'chemical',
            ): self.side_effect_proj(
                self.side_effect_emb(
                    rev_chem_ids
                )
            ),
            (
                'drugs',
                'rev_targets',
                'patient',
            ): rev_demo_edge_attr,
        }

        for conv in self.convs:
            updated = conv(
                x_dict,
                batch.edge_index_dict,
                edge_attr_dict,
            )

            x_dict = {
                node_type: self.dropout(
                    F.relu(
                        updated[node_type]
                        + x_dict[node_type]
                    )
                )
                for node_type in updated
            }

        num_seed_patients = (
            batch['patient']
            .batch_size
        )

        patient_emb = (
            x_dict['patient']
            [:num_seed_patients]
        )

        outcome_emb = (
            self.outcome_emb.weight
        )

        batch_size = patient_emb.size(0)
        num_outcomes = outcome_emb.size(0)

        patient_expanded = (
            patient_emb
            .unsqueeze(1)
            .expand(
                batch_size,
                num_outcomes,
                self.hidden_channels,
            )
        )

        outcome_expanded = (
            outcome_emb
            .unsqueeze(0)
            .expand(
                batch_size,
                num_outcomes,
                self.hidden_channels,
            )
        )

        edge_input = torch.cat(
            [
                patient_expanded,
                outcome_expanded,
            ],
            dim=-1,
        )

        return (
            self.edge_predictor(
                edge_input
            )
            .squeeze(-1)
        )


def main():
    cwd = Path.cwd()

    subgraph_path = (
        cwd
        / 'output_csv'
        / 'graph_data'
        / 'single_patient_subgraph.pt'
    )
    embedding_path = (
        cwd
        / 'output_csv'
        / 'pre-embeddings'
        / 'pre_embeddings.pt'
    )
    checkpoint_path = (
        cwd
        / 'gnn_multilabel_outcome_best.pth'
    )

    payload = safe_torch_load(
        subgraph_path
    )
    emb_map = safe_torch_load(
        embedding_path
    )
    checkpoint = safe_torch_load(
        checkpoint_path
    )

    batch = payload['batch']

    checkpoint_codes = checkpoint.get(
        'outcome_codes',
        OUTCOME_CODES,
    )

    if list(checkpoint_codes) != OUTCOME_CODES:
        raise ValueError(
            'Checkpoint outcome order does not match '
            '[DE, LT, HO, DS, CA, RI, OT].'
        )

    state = checkpoint[
        'model_state_dict'
    ]

    hidden_channels = int(
        state[
            'outcome_emb.weight'
        ].size(1)
    )

    num_outcomes = int(
        state[
            'outcome_emb.weight'
        ].size(0)
    )

    num_genes = int(
        state[
            'gene_emb.weight'
        ].size(0)
    )

    num_mfr = int(
        state[
            'mfr_emb.weight'
        ].size(0)
    )

    num_occr = int(
        state[
            'occr_emb.weight'
        ].size(0)
    )

    num_sex = int(
        state[
            'sex_emb.weight'
        ].size(0)
    )

    patient_num_dim = int(
        batch['patient']
        .numerical
        .size(1)
    )

    side_effect_tensor = torch.stack(
        list(
            emb_map[
                'side_effect'
            ].values()
        )
    )

    side_effect_emb = (
        nn.Embedding.from_pretrained(
            side_effect_tensor,
            freeze=True,
        )
    )

    placeholder_mean = torch.zeros(
        patient_num_dim,
        dtype=torch.float32,
    )
    placeholder_std = torch.ones(
        patient_num_dim,
        dtype=torch.float32,
    )

    num_layers = int(
        checkpoint.get(
            'num_gnn_layers',
            3,
        )
    )

    model = advGNN(
        hidden_channels=hidden_channels,
        chemical_feat_dim=(
            batch['chemical']
            .x
            .size(1)
        ),
        demo_edge_feat_dim=(
            batch[
                'drugs',
                'rev_targets',
                'patient',
            ]
            .edge_attr
            .size(1)
        ),
        num_genes=num_genes,
        side_effect_emb=side_effect_emb,
        num_mfr=num_mfr,
        num_occr=num_occr,
        num_sex=num_sex,
        patient_num_dim=patient_num_dim,
        num_outcomes=num_outcomes,
        patient_num_mean=placeholder_mean,
        patient_num_std=placeholder_std,
        num_layers=num_layers,
        dropout=DROPOUT,
    )

    model.load_state_dict(state)

    device = torch.device(
        'cuda'
        if torch.cuda.is_available()
        else 'cpu'
    )

    model = model.to(device)
    batch = batch.to(device)

    model.eval()

    with torch.no_grad():
        logits = model(batch)
        probabilities = (
            torch.sigmoid(logits)
            .squeeze(0)
            .cpu()
            .numpy()
        )

    thresholds_dict = checkpoint.get(
        'decision_thresholds',
        {},
    )

    thresholds = np.array([
        float(
            thresholds_dict.get(
                code,
                0.5,
            )
        )
        for code in OUTCOME_CODES
    ])

    predictions = (
        probabilities
        >= thresholds
    ).astype(np.int64)

    actual = (
        payload['actual_target']
        .cpu()
        .numpy()
        .astype(np.int64)
    )

    primaryid = payload[
        'primaryid'
    ]

    print(
        f'Patient: {primaryid}'
    )
    print(
        f'Global patient index: '
        f'{payload["global_patient_index"]}'
    )
    print(
        f'Device: {device}'
    )

    print(
        '\nActual positive outcomes: '
        + (
            ', '.join(
                code
                for code, value
                in zip(
                    OUTCOME_CODES,
                    actual,
                )
                if value == 1
            )
            or 'None'
        )
    )

    print(
        'Predicted positive outcomes: '
        + (
            ', '.join(
                code
                for code, value
                in zip(
                    OUTCOME_CODES,
                    predictions,
                )
                if value == 1
            )
            or 'None'
        )
    )

    print(
        '\nOutcome predictions:'
    )

    print(
        f"{'Outcome':<8}"
        f"{'Probability':>14}"
        f"{'Threshold':>12}"
        f"{'Predicted':>12}"
        f"{'Actual':>10}"
        f"{'Correct':>10}"
    )

    for (
        code,
        probability,
        threshold,
        predicted,
        true_value,
    ) in zip(
        OUTCOME_CODES,
        probabilities,
        thresholds,
        predictions,
        actual,
    ):
        correct = (
            predicted == true_value
        )

        print(
            f'{code:<8}'
            f'{probability:>14.4f}'
            f'{threshold:>12.4f}'
            f'{("YES" if predicted else "NO"):>12}'
            f'{("YES" if true_value else "NO"):>10}'
            f'{("YES" if correct else "NO"):>10}'
        )

    exact_match = bool(
        np.array_equal(
            predictions,
            actual,
        )
    )

    correct_labels = int(
        (predictions == actual)
        .sum()
    )

    print(
        f'\nCorrect binary decisions: '
        f'{correct_labels}/'
        f'{len(OUTCOME_CODES)}'
    )

    print(
        'Exact outcome-set match: '
        + (
            'YES'
            if exact_match
            else 'NO'
        )
    )

    ranked = sorted(
        zip(
            OUTCOME_CODES,
            probabilities,
        ),
        key=lambda item: item[1],
        reverse=True,
    )

    print(
        '\nRanked outcome probabilities:'
    )

    for rank, (
        code,
        probability,
    ) in enumerate(
        ranked,
        start=1,
    ):
        print(
            f'  {rank}. '
            f'{code}: '
            f'{probability:.4f}'
        )


if __name__ == '__main__':
    main()
