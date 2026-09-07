from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import NeighborLoader


SEED = 42
NUM_GNN_LAYERS = 3
NUM_NEIGHBORS = [10, 7, 5]

OUTCOME_CODES = [
    'DE',
    'LT',
    'HO',
    'DS',
    'CA',
    'RI',
    'OT',
]

PATIENT_PRIMARYID = '15Q1_100128784'


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


def sanitize_fixed_tensor(name, tensor):
    if not torch.is_tensor(tensor):
        return tensor

    if torch.is_floating_point(tensor):
        bad = ~torch.isfinite(tensor)
        count = int(bad.sum().item())

        if count:
            print(
                f'WARNING: {name} contains {count} NaN/Inf values; '
                'replacing them with 0.'
            )
            tensor = torch.nan_to_num(
                tensor,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )

    return tensor.detach()


def prepare_demo(demo):
    missing_columns = [
        col
        for col in OUTCOME_CODES
        if col not in demo.columns
    ]

    if missing_columns:
        raise ValueError(
            f'Missing outcome columns: {missing_columns}'
        )

    if 'has_outcome_label' in demo.columns:
        if demo['has_outcome_label'].dtype == object:
            parsed = (
                demo['has_outcome_label']
                .astype(str)
                .str.strip()
                .str.upper()
                .map({
                    'TRUE': True,
                    'FALSE': False,
                    '1': True,
                    '0': False,
                })
            )

            if parsed.isna().any():
                bad_values = sorted(
                    demo.loc[
                        parsed.isna(),
                        'has_outcome_label',
                    ]
                    .astype(str)
                    .unique()
                    .tolist()
                )
                raise ValueError(
                    'Could not parse has_outcome_label values: '
                    f'{bad_values}'
                )

            demo['has_outcome_label'] = parsed

        demo = demo.loc[
            demo['has_outcome_label']
            .fillna(False)
            .astype(bool)
        ].copy()

    for col in OUTCOME_CODES:
        demo[col] = pd.to_numeric(
            demo[col],
            errors='coerce',
        )

    valid_target_mask = (
        demo[OUTCOME_CODES]
        .notna()
        .all(axis=1)
    )

    demo = demo.loc[
        valid_target_mask
    ].copy()

    if not (
        demo[OUTCOME_CODES]
        .isin([0, 1])
        .all()
        .all()
    ):
        raise ValueError(
            'Outcome columns must contain only 0/1 values.'
        )

    return demo


def prepare_graph_for_inference(data):
    data['patient'].num_nodes = int(
        data['patient'].y.size(0)
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
    data['patient'].y = sanitize_fixed_tensor(
        'patient.y',
        data['patient'].y.float(),
    )

    disease_edge = (
        'chemical',
        'targets',
        'disease_class',
    )
    reverse_disease_edge = (
        'disease_class',
        'rev_targets',
        'chemical',
    )

    if reverse_disease_edge not in data.edge_types:
        data[reverse_disease_edge].edge_index = (
            data[disease_edge]
            .edge_index
            .flip(0)
        )

    patient_to_drug_edge = (
        'patient',
        'targets',
        'drugs',
    )
    reverse_patient_drug_edge = (
        'drugs',
        'rev_targets',
        'patient',
    )

    if patient_to_drug_edge in data.edge_types:
        edge_attr = sanitize_fixed_tensor(
            'patient->drugs.edge_attr',
            data[patient_to_drug_edge].edge_attr,
        )

        data[
            reverse_patient_drug_edge
        ].edge_attr = edge_attr

        del data[patient_to_drug_edge]

    elif (
        reverse_patient_drug_edge in data.edge_types
        and 'edge_attr'
        in data[reverse_patient_drug_edge]
    ):
        data[
            reverse_patient_drug_edge
        ].edge_attr = sanitize_fixed_tensor(
            'drugs->patient.edge_attr',
            data[
                reverse_patient_drug_edge
            ].edge_attr,
        )
    else:
        raise ValueError(
            'Could not find patient-drug edge attributes.'
        )

    return data


def main():
    set_seed()
    cwd = Path.cwd()

    graph_path = (
        cwd
        / 'output_csv'
        / 'graph_data'
        / 'graph_data.pt'
    )
    demo_path = (
        cwd
        / 'output_csv'
        / 'demo_clean'
        / 'demo_clean.csv'
    )

    data = safe_torch_load(graph_path)
    demo = pd.read_csv(demo_path)

    demo = prepare_demo(demo)

    if len(demo) != data['patient'].y.size(0):
        raise ValueError(
            'The filtered demo row count does not match the '
            'patient count in graph_data.pt. Rebuild the graph '
            'or verify that demo_clean.csv is the same file used '
            'to create it.'
        )

    primaryids = (
        demo['primaryid']
        .astype(str)
        .to_numpy()
    )

    matches = np.flatnonzero(
        primaryids == str(PATIENT_PRIMARYID)
    )

    if len(matches) == 0:
        raise ValueError(
            f'Patient {PATIENT_PRIMARYID} was not found '
            'in the supervised graph.'
        )

    if len(matches) > 1:
        raise ValueError(
            f'Patient {PATIENT_PRIMARYID} occurs '
            f'{len(matches)} times after filtering.'
        )

    patient_idx = int(matches[0])
    patient_row = demo.iloc[patient_idx]

    csv_target = torch.tensor(
        patient_row[OUTCOME_CODES]
        .to_numpy(dtype=np.float32),
        dtype=torch.float32,
    )

    graph_target = (
        data['patient']
        .y[patient_idx]
        .float()
        .cpu()
    )

    if not torch.equal(
        csv_target,
        graph_target,
    ):
        raise ValueError(
            'The patient outcome vector in demo_clean.csv '
            'does not match graph_data.pt. This indicates '
            'that the files were created from different '
            'preprocessing runs.'
        )

    actual_outcomes = [
        code
        for code, value
        in zip(
            OUTCOME_CODES,
            graph_target.tolist(),
        )
        if value == 1
    ]

    print(
        f'Patient: {PATIENT_PRIMARYID}'
    )
    print(
        f'Global patient node index: {patient_idx}'
    )
    print(
        'Actual outcome vector: '
        f'{graph_target.int().tolist()}'
    )
    print(
        'Actual positive outcomes: '
        + (
            ', '.join(actual_outcomes)
            if actual_outcomes
            else 'None'
        )
    )

    data = prepare_graph_for_inference(data)

    if len(NUM_NEIGHBORS) != NUM_GNN_LAYERS:
        raise ValueError(
            'NUM_NEIGHBORS must have one entry '
            'per GNN layer.'
        )

    neighbor_spec = {
        edge_type: NUM_NEIGHBORS
        for edge_type in data.edge_types
    }

    input_nodes = torch.tensor(
        [patient_idx],
        dtype=torch.long,
    )

    loader = NeighborLoader(
        data,
        num_neighbors=neighbor_spec,
        input_nodes=(
            'patient',
            input_nodes,
        ),
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    batch = next(iter(loader))

    if batch['patient'].batch_size != 1:
        raise RuntimeError(
            'Expected exactly one seed patient.'
        )

    sampled_global_id = int(
        batch['patient'].n_id[0].item()
    )

    if sampled_global_id != patient_idx:
        raise RuntimeError(
            'NeighborLoader seed patient does not '
            'match the requested patient index.'
        )

    payload = {
        'batch': batch,
        'primaryid': str(PATIENT_PRIMARYID),
        'global_patient_index': patient_idx,
        'outcome_codes': OUTCOME_CODES,
        'actual_target': graph_target,
        'actual_outcomes': actual_outcomes,
        'num_neighbors': NUM_NEIGHBORS,
        'seed': SEED,
    }

    output_dir = (
        cwd
        / 'output_csv'
        / 'graph_data'
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir
        / 'single_patient_subgraph.pt'
    )

    torch.save(
        payload,
        output_path,
    )

    print('\nSampled node counts:')
    for node_type in batch.node_types:
        print(
            f'  {node_type}: '
            f'{batch[node_type].num_nodes}'
        )

    print('\nSampled edge counts:')
    for edge_type in batch.edge_types:
        print(
            f'  {edge_type}: '
            f'{batch[edge_type].edge_index.size(1)}'
        )

    print(
        f'\nSaved single-patient sampled graph to: '
        f'{output_path}'
    )


if __name__ == '__main__':
    main()
