from pathlib import Path
import ast
import math
import random
import re
import string
import zipfile

import numpy as np
import pandas as pd
import requests
import torch
from torch_geometric.loader import NeighborLoader


SEED = 42
NUM_GNN_LAYERS = 3
NUM_NEIGHBORS = [10, 7, 5]

OUTCOME_CODES = ['DE', 'LT', 'HO', 'DS', 'CA', 'RI', 'OT']

QUARTER = '2026Q2'
FDA_ARCHIVE_URL = 'https://fis.fda.gov/content/Exports/faers_ascii_2026q2.zip'

# Set to None to take any compatible serious report.
# DE is useful here because your full model was only moderate on death.
REQUIRE_OUTCOME = 'DE'

MIN_DRUGS = 1
MAX_DRUGS = 10


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


def norm_text(x):
    if x is None or pd.isna(x):
        return ''
    return str(x).strip()


def normalise_name(name):
    name = norm_text(name).lower()
    keep = '-()'
    name = name.translate(
        str.maketrans(
            '',
            '',
            ''.join(
                c
                for c in string.punctuation
                if c not in keep
            ),
        )
    )
    return name


def take_first(names):
    names = norm_text(names)
    parts = re.split(r'w/|/|\+| with ', names)
    return parts[0].strip()


def normalise_drug_name(name):
    return normalise_name(take_first(name))


def parse_bool_column(series):
    if series.dtype != object:
        return series.fillna(False).astype(bool)

    parsed = (
        series.astype(str)
        .str.strip()
        .str.upper()
        .map({
            'TRUE': True,
            'FALSE': False,
            '1': True,
            '0': False,
        })
    )

    return parsed.fillna(False)


def prepare_training_demo(demo):
    missing = [
        col
        for col in OUTCOME_CODES
        if col not in demo.columns
    ]
    if missing:
        raise ValueError(
            f'Training demo is missing outcome columns: {missing}'
        )

    if 'has_outcome_label' in demo.columns:
        keep = parse_bool_column(
            demo['has_outcome_label']
        )
        demo = demo.loc[keep].copy()

    for col in OUTCOME_CODES:
        demo[col] = pd.to_numeric(
            demo[col],
            errors='coerce',
        )

    demo = demo.loc[
        demo[OUTCOME_CODES]
        .notna()
        .all(axis=1)
    ].copy()

    return demo


def prepare_training_drug_df(drug_df, valid_primaryids):
    drug_df = drug_df.loc[
        drug_df['primaryid'].isin(
            valid_primaryids
        )
    ].copy()

    drug_df['route'] = (
        drug_df['route']
        .astype('string')
        .str.lower()
    )

    drug_df['dose_unit'] = (
        drug_df['dose_unit']
        .fillna('UNKNOWN')
    )

    drug_df['start_dt'] = pd.to_datetime(
        drug_df['start_dt'],
        errors='coerce',
    )
    drug_df['end_dt'] = pd.to_datetime(
        drug_df['end_dt'],
        errors='coerce',
    )

    drug_df['duration_days'] = (
        drug_df['end_dt']
        - drug_df['start_dt']
    ).dt.days

    drug_df = drug_df.dropna(
        subset=['drugname']
    ).copy()

    drug_df['drugname'] = (
        drug_df['drugname']
        .apply(norm_text)
    )

    return drug_df


def build_string_index_map(
    values,
    indices,
):
    result = {}

    for value, index in zip(
        values,
        indices,
    ):
        key = norm_text(value)

        if key not in result:
            result[key] = int(index)

    return result


def build_vector_map(
    values,
    vectors,
    normalizer=lambda x: norm_text(x),
):
    result = {}

    for value, vector in zip(
        values,
        vectors,
    ):
        key = normalizer(value)

        if key and key not in result:
            result[key] = (
                vector.detach()
                .cpu()
                .clone()
            )

    return result


def sanitize_graph(data):
    for node_type in [
        'chemical',
        'drugs',
        'indi_pt',
        'disease_class',
    ]:
        data[node_type].x = (
            torch.nan_to_num(
                data[node_type].x.float(),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            .detach()
        )

    data['patient'].numerical = (
        torch.nan_to_num(
            data['patient'].numerical.float(),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        .detach()
    )

    data['patient'].y = (
        data['patient'].y
        .float()
        .detach()
    )

    return data


def ensure_latest_quarter_files(cwd):
    base_dir = (
        cwd
        / 'external_faers'
        / QUARTER.lower()
    )
    base_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    zip_path = (
        base_dir
        / 'faers_ascii_2026q2.zip'
    )

    if not zip_path.exists():
        print(
            'Downloading official FDA '
            '2026 Q2 FAERS ASCII archive...'
        )

        response = requests.get(
            FDA_ARCHIVE_URL,
            timeout=180,
        )
        response.raise_for_status()

        zip_path.write_bytes(
            response.content
        )

    extract_dir = base_dir / 'ascii'

    if not extract_dir.exists():
        print('Extracting 2026 Q2 archive...')

        extract_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        with zipfile.ZipFile(
            zip_path,
            'r',
        ) as zf:
            zf.extractall(
                extract_dir
            )

    return extract_dir


def find_quarter_file(
    root,
    prefix,
):
    matches = [
        path
        for path in root.rglob('*')
        if path.is_file()
        and path.name.upper().startswith(
            prefix.upper()
        )
        and path.suffix.lower() == '.txt'
    ]

    if not matches:
        raise FileNotFoundError(
            f'Could not find {prefix}*.txt '
            f'under {root}'
        )

    return sorted(matches)[0]


def read_faers_file(path):
    return pd.read_csv(
        path,
        sep='$',
        dtype=str,
        encoding='latin1',
        low_memory=False,
        on_bad_lines='skip',
    )


def parse_faers_date(value):
    text = re.sub(
        r'[^0-9]',
        '',
        norm_text(value),
    )

    if len(text) >= 8:
        return pd.to_datetime(
            text[:8],
            format='%Y%m%d',
            errors='coerce',
        )

    if len(text) == 6:
        return pd.to_datetime(
            text + '01',
            format='%Y%m%d',
            errors='coerce',
        )

    if len(text) == 4:
        return pd.to_datetime(
            text + '0101',
            format='%Y%m%d',
            errors='coerce',
        )

    return pd.NaT


def age_to_years(age, age_cod):
    age = pd.to_numeric(
        age,
        errors='coerce',
    )

    if pd.isna(age):
        return None

    unit_to_years = {
        'DY': 1 / 365,
        'HR': 1 / (365 * 24),
        'WK': 1 / 52,
        'MON': 1 / 12,
        'YR': 1,
        'DEC': 10,
    }

    code = norm_text(
        age_cod
    ).upper()

    if code not in unit_to_years:
        return None

    return float(
        age * unit_to_years[code]
    )


def reconstruct_training_maps(
    data,
    training_demo,
    training_drug_df,
):
    if len(training_demo) != data['patient'].y.size(0):
        raise ValueError(
            'Filtered demo_clean.csv does not '
            'match graph_data.pt patient count.'
        )

    cat_index = (
        data['patient']
        .cat_index
        .cpu()
    )

    demo_for_maps = (
        training_demo
        .copy()
    )

    valid_sex = {
        'F',
        'M',
        'UNK',
    }

    demo_for_maps['sex'] = (
        demo_for_maps['sex']
        .fillna('UNK')
        .apply(
            lambda x:
            x
            if x in valid_sex
            else 'UNK'
        )
    )

    demo_for_maps['mfr_sndr'] = (
        demo_for_maps['mfr_sndr']
        .fillna('UNKNOWN')
    )

    demo_for_maps['occr_country'] = (
        demo_for_maps['occr_country']
        .fillna('UNKNOWN')
    )

    mfr_map = build_string_index_map(
        demo_for_maps['mfr_sndr'],
        cat_index[:, 0],
    )

    occr_map = build_string_index_map(
        demo_for_maps['occr_country'],
        cat_index[:, 1],
    )

    sex_map = build_string_index_map(
        demo_for_maps['sex'],
        cat_index[:, 2],
    )

    unique_drug = (
        training_drug_df
        .drop_duplicates(
            subset='drugname'
        )
    )

    drug_link_map = {
        drug: idx
        for idx, drug
        in enumerate(
            unique_drug['drugname']
        )
    }

    forward_edge = (
        'patient',
        'targets',
        'drugs',
    )

    if forward_edge not in data.edge_types:
        raise ValueError(
            'graph_data.pt must still contain '
            'patient->drug edges.'
        )

    edge_attr = (
        data[forward_edge]
        .edge_attr
        .detach()
        .cpu()
    )

    if len(training_drug_df) != edge_attr.size(0):
        raise ValueError(
            'Training drug row count does not '
            'match patient->drug edge attributes.'
        )

    if edge_attr.size(1) != 29:
        raise ValueError(
            'Expected 29 patient->drug edge '
            f'features, found {edge_attr.size(1)}.'
        )

    role_vec_map = build_vector_map(
        training_drug_df['role_cod'],
        edge_attr[:, 0:4],
        lambda x: norm_text(x).upper(),
    )

    route_vec_map = build_vector_map(
        training_drug_df['route'],
        edge_attr[:, 4:12],
        lambda x: norm_text(x).lower(),
    )

    dechal_vec_map = build_vector_map(
        training_drug_df['dechal'],
        edge_attr[:, 12:16],
        lambda x: norm_text(x).upper(),
    )

    rechal_vec_map = build_vector_map(
        training_drug_df['rechal'],
        edge_attr[:, 16:20],
        lambda x: norm_text(x).upper(),
    )

    dose_unit_vec_map = build_vector_map(
        training_drug_df['dose_unit'],
        edge_attr[:, 20:26],
        lambda x: norm_text(x).upper(),
    )

    return {
        'mfr': mfr_map,
        'occr': occr_map,
        'sex': sex_map,
        'drug_link': drug_link_map,
        'role_vec': role_vec_map,
        'route_vec': route_vec_map,
        'dechal_vec': dechal_vec_map,
        'rechal_vec': rechal_vec_map,
        'dose_unit_vec': dose_unit_vec_map,
    }


def fallback_index(
    mapping,
    value,
    fallback='UNKNOWN',
):
    key = norm_text(value)

    if key in mapping:
        return mapping[key]

    if fallback in mapping:
        return mapping[fallback]

    raise KeyError(
        f'Value {value!r} is unseen and '
        f'fallback {fallback!r} is unavailable.'
    )


def prepare_external_drugs(
    rows,
    therapy_rows,
    maps,
):
    rows = rows.copy()

    rows['drug_seq'] = (
        rows['drug_seq']
        .astype(str)
    )

    duration_by_seq = {}

    if (
        therapy_rows is not None
        and not therapy_rows.empty
        and 'dsg_drug_seq'
        in therapy_rows.columns
    ):
        for _, ther in (
            therapy_rows
            .drop_duplicates(
                subset='dsg_drug_seq'
            )
            .iterrows()
        ):
            seq = str(
                ther['dsg_drug_seq']
            )

            start = parse_faers_date(
                ther.get(
                    'start_dt',
                    '',
                )
            )
            end = parse_faers_date(
                ther.get(
                    'end_dt',
                    '',
                )
            )

            if (
                pd.notna(start)
                and pd.notna(end)
            ):
                duration = (
                    end - start
                ).days
            else:
                duration = 0

            duration_by_seq[
                seq
            ] = float(duration)

    prepared = []

    for _, row in rows.iterrows():
        drugname = normalise_drug_name(
            row.get(
                'drugname',
                '',
            )
        )

        if (
            not drugname
            or drugname
            not in maps['drug_link']
        ):
            return None

        role = norm_text(
            row.get(
                'role_cod',
                '',
            )
        ).upper()

        route = norm_text(
            row.get(
                'route',
                '',
            )
        ).lower()

        dechal = norm_text(
            row.get(
                'dechal',
                '',
            )
        ).upper() or 'U'

        rechal = norm_text(
            row.get(
                'rechal',
                '',
            )
        ).upper() or 'U'

        dose_unit = norm_text(
            row.get(
                'dose_unit',
                '',
            )
        ).upper() or 'UNKNOWN'

        if role not in maps['role_vec']:
            return None

        if route not in maps['route_vec']:
            return None

        if dechal not in maps['dechal_vec']:
            return None

        if rechal not in maps['rechal_vec']:
            return None

        if dose_unit not in maps['dose_unit_vec']:
            return None

        dose_amt_raw = pd.to_numeric(
            row.get(
                'dose_amt',
                np.nan,
            ),
            errors='coerce',
        )

        dose_missing = float(
            pd.isna(
                dose_amt_raw
            )
        )

        dose_amt = (
            0.0
            if pd.isna(dose_amt_raw)
            else float(dose_amt_raw)
        )

        seq = str(
            row.get(
                'drug_seq',
                '',
            )
        )

        duration = float(
            duration_by_seq.get(
                seq,
                0.0,
            )
        )

        edge_attr = torch.cat([
            maps['role_vec'][role],
            maps['route_vec'][route],
            maps['dechal_vec'][dechal],
            maps['rechal_vec'][rechal],
            maps['dose_unit_vec'][dose_unit],
            torch.tensor(
                [
                    dose_amt,
                    duration,
                    dose_missing,
                ],
                dtype=torch.float32,
            ),
        ])

        prepared.append({
            'drugname': drugname,
            'drug_idx': maps['drug_link'][drugname],
            'edge_attr': edge_attr,
            'role_cod': role,
            'route': route,
            'dechal': dechal,
            'rechal': rechal,
            'dose_amt': dose_amt,
            'dose_unit': dose_unit,
            'duration_days': duration,
        })

    return prepared


def choose_external_patient(
    demo_2026,
    drug_2026,
    outc_2026,
    ther_2026,
    maps,
):
    for df in [
        demo_2026,
        drug_2026,
        outc_2026,
        ther_2026,
    ]:
        if 'primaryid' in df.columns:
            df['primaryid'] = (
                df['primaryid']
                .astype(str)
            )

        if 'caseid' in df.columns:
            df['caseid'] = (
                df['caseid']
                .astype(str)
            )

    outcomes = (
        outc_2026[
            outc_2026[
                'outc_cod'
            ].isin(
                OUTCOME_CODES
            )
        ]
        .groupby(
            'primaryid'
        )['outc_cod']
        .agg(
            lambda values:
            sorted(
                set(values),
                key=OUTCOME_CODES.index,
            )
        )
        .to_dict()
    )

    candidates = (
        demo_2026[
            demo_2026[
                'primaryid'
            ].isin(
                outcomes.keys()
            )
        ]
        .copy()
    )

    candidates['_fda_sort'] = (
        candidates.get(
            'fda_dt',
            ''
        )
        .astype(str)
        .str.replace(
            r'[^0-9]',
            '',
            regex=True,
        )
    )

    candidates['_primary_sort'] = pd.to_numeric(
        candidates['primaryid'],
        errors='coerce',
    ).fillna(-1)

    candidates = candidates.sort_values(
        [
            '_fda_sort',
            '_primary_sort',
        ],
        ascending=False,
    )

    for _, demo_row in candidates.iterrows():
        primaryid = str(
            demo_row['primaryid']
        )

        patient_outcomes = (
            outcomes.get(
                primaryid,
                [],
            )
        )

        if (
            REQUIRE_OUTCOME is not None
            and REQUIRE_OUTCOME
            not in patient_outcomes
        ):
            continue

        age_years = age_to_years(
            demo_row.get(
                'age',
                None,
            ),
            demo_row.get(
                'age_cod',
                None,
            ),
        )

        if age_years is None:
            continue

        event_date = parse_faers_date(
            demo_row.get(
                'event_dt',
                '',
            )
        )

        if pd.isna(event_date):
            continue

        patient_drugs = drug_2026.loc[
            drug_2026[
                'primaryid'
            ] == primaryid
        ].copy()

        if not (
            MIN_DRUGS
            <= len(patient_drugs)
            <= MAX_DRUGS
        ):
            continue

        patient_therapy = ther_2026.loc[
            ther_2026[
                'primaryid'
            ] == primaryid
        ].copy()

        prepared_drugs = prepare_external_drugs(
            patient_drugs,
            patient_therapy,
            maps,
        )

        if not prepared_drugs:
            continue

        target = torch.tensor(
            [
                1.0
                if code
                in patient_outcomes
                else 0.0
                for code
                in OUTCOME_CODES
            ],
            dtype=torch.float32,
        )

        return {
            'primaryid': primaryid,
            'caseid': str(
                demo_row.get(
                    'caseid',
                    '',
                )
            ),
            'mfr_sndr': norm_text(
                demo_row.get(
                    'mfr_sndr',
                    'UNKNOWN',
                )
            ) or 'UNKNOWN',
            'occr_country': norm_text(
                demo_row.get(
                    'occr_country',
                    'UNKNOWN',
                )
            ) or 'UNKNOWN',
            'sex': norm_text(
                demo_row.get(
                    'sex',
                    'UNK',
                )
            ).upper() or 'UNK',
            'age_years': float(
                age_years
            ),
            'event_year': int(
                event_date.year
            ),
            'outcomes': patient_outcomes,
            'target': target,
            'drugs': prepared_drugs,
        }

    raise RuntimeError(
        'No compatible 2026 Q2 patient was found. '
        'Try setting REQUIRE_OUTCOME = None or increasing MAX_DRUGS.'
    )


def append_and_sample_patient(
    data,
    patient,
    maps,
):
    old_num_patients = int(
        data['patient']
        .y
        .size(0)
    )

    mfr_idx = fallback_index(
        maps['mfr'],
        patient['mfr_sndr'],
        'UNKNOWN',
    )

    occr_idx = fallback_index(
        maps['occr'],
        patient['occr_country'],
        'UNKNOWN',
    )

    sex_value = (
        patient['sex']
        if patient['sex']
        in maps['sex']
        else 'UNK'
    )

    sex_idx = fallback_index(
        maps['sex'],
        sex_value,
        'UNK',
    )

    new_cat = torch.tensor(
        [[
            mfr_idx,
            occr_idx,
            sex_idx,
        ]],
        dtype=torch.long,
    )

    new_num = torch.tensor(
        [[
            patient['event_year'],
            patient['age_years'],
        ]],
        dtype=torch.float32,
    )

    data['patient'].cat_index = torch.cat([
        data['patient']
        .cat_index
        .long(),
        new_cat,
    ], dim=0)

    data['patient'].numerical = torch.cat([
        data['patient']
        .numerical
        .float(),
        new_num,
    ], dim=0)

    data['patient'].y = torch.cat([
        data['patient']
        .y
        .float(),
        patient['target']
        .unsqueeze(0),
    ], dim=0)

    new_patient_idx = old_num_patients

    new_src = torch.full(
        (
            len(
                patient['drugs']
            ),
        ),
        new_patient_idx,
        dtype=torch.long,
    )

    new_dst = torch.tensor(
        [
            drug['drug_idx']
            for drug
            in patient['drugs']
        ],
        dtype=torch.long,
    )

    new_edges = torch.stack([
        new_src,
        new_dst,
    ], dim=0)

    new_edge_attr = torch.stack([
        drug['edge_attr']
        for drug
        in patient['drugs']
    ]).float()

    forward_edge = (
        'patient',
        'targets',
        'drugs',
    )
    reverse_edge = (
        'drugs',
        'rev_targets',
        'patient',
    )

    data[forward_edge].edge_index = torch.cat([
        data[forward_edge]
        .edge_index
        .long(),
        new_edges,
    ], dim=1)

    data[forward_edge].edge_attr = torch.cat([
        data[forward_edge]
        .edge_attr
        .float()
        .detach(),
        new_edge_attr,
    ], dim=0)

    data[reverse_edge].edge_index = torch.cat([
        data[reverse_edge]
        .edge_index
        .long(),
        new_edges.flip(0),
    ], dim=1)

    data[reverse_edge].edge_attr = (
        data[forward_edge]
        .edge_attr
        .detach()
    )

    data['patient'].num_nodes = (
        old_num_patients + 1
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
        data[
            reverse_disease_edge
        ].edge_index = (
            data[disease_edge]
            .edge_index
            .flip(0)
        )

    del data[forward_edge]

    neighbor_spec = {
        edge_type: NUM_NEIGHBORS
        for edge_type
        in data.edge_types
    }

    loader = NeighborLoader(
        data,
        num_neighbors=neighbor_spec,
        input_nodes=(
            'patient',
            torch.tensor(
                [new_patient_idx],
                dtype=torch.long,
            ),
        ),
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    batch = next(
        iter(loader)
    )

    return (
        batch,
        new_patient_idx,
    )


def main():
    set_seed()
    cwd = Path.cwd()

    graph_path = (
        cwd
        / 'output_csv'
        / 'graph_data'
        / 'graph_data.pt'
    )
    training_demo_path = (
        cwd
        / 'output_csv'
        / 'demo_clean'
        / 'demo_clean.csv'
    )
    training_drug_path = (
        cwd
        / 'output_csv'
        / 'drug_cid'
        / 'drug_cid_attached.csv'
    )

    data = safe_torch_load(
        graph_path
    )
    data = sanitize_graph(
        data
    )

    training_demo = pd.read_csv(
        training_demo_path
    )
    training_drug_df = pd.read_csv(
        training_drug_path
    )

    training_demo = prepare_training_demo(
        training_demo
    )

    valid_primaryids = set(
        training_demo[
            'primaryid'
        ]
    )

    training_drug_df = (
        prepare_training_drug_df(
            training_drug_df,
            valid_primaryids,
        )
    )

    maps = reconstruct_training_maps(
        data,
        training_demo,
        training_drug_df,
    )

    quarter_dir = ensure_latest_quarter_files(
        cwd
    )

    demo_file = find_quarter_file(
        quarter_dir,
        'DEMO',
    )
    drug_file = find_quarter_file(
        quarter_dir,
        'DRUG',
    )
    outc_file = find_quarter_file(
        quarter_dir,
        'OUTC',
    )
    ther_file = find_quarter_file(
        quarter_dir,
        'THER',
    )

    print('Reading 2026 Q2 FAERS files...')

    demo_2026 = read_faers_file(
        demo_file
    )
    drug_2026 = read_faers_file(
        drug_file
    )
    outc_2026 = read_faers_file(
        outc_file
    )
    ther_2026 = read_faers_file(
        ther_file
    )

    patient = choose_external_patient(
        demo_2026,
        drug_2026,
        outc_2026,
        ther_2026,
        maps,
    )

    print('\nSelected external patient:')
    print(
        f'  Quarter: {QUARTER}'
    )
    print(
        f'  primaryid: '
        f'{patient["primaryid"]}'
    )
    print(
        f'  caseid: '
        f'{patient["caseid"]}'
    )
    print(
        f'  age: '
        f'{patient["age_years"]:.2f} years'
    )
    print(
        f'  sex: '
        f'{patient["sex"]}'
    )
    print(
        f'  occurrence country: '
        f'{patient["occr_country"]}'
    )
    print(
        f'  manufacturer/sender: '
        f'{patient["mfr_sndr"]}'
    )
    print(
        '  actual outcomes: '
        + ', '.join(
            patient['outcomes']
        )
    )

    print('  drugs:')
    for drug in patient['drugs']:
        print(
            f'    - {drug["drugname"]} '
            f'[{drug["role_cod"]}] '
            f'route={drug["route"] or "unknown"} '
            f'dose={drug["dose_amt"]} '
            f'{drug["dose_unit"]}'
        )

    batch, new_patient_idx = (
        append_and_sample_patient(
            data,
            patient,
            maps,
        )
    )

    payload = {
        'batch': batch,
        'primaryid': patient[
            'primaryid'
        ],
        'caseid': patient[
            'caseid'
        ],
        'quarter': QUARTER,
        'global_patient_index': (
            new_patient_idx
        ),
        'outcome_codes': OUTCOME_CODES,
        'actual_target': patient[
            'target'
        ],
        'actual_outcomes': patient[
            'outcomes'
        ],
        'external_patient': True,
        'patient_details': {
            key: value
            for key, value
            in patient.items()
            if key
            not in {
                'target',
                'drugs',
            }
        },
        'drug_details': patient[
            'drugs'
        ],
        'num_neighbors': (
            NUM_NEIGHBORS
        ),
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

    print(
        '\nThis patient is external to '
        'the 2015-2025 training period.'
    )

    print(
        f'Saved inference graph to: '
        f'{output_path}'
    )

    print(
        '\nNow run: '
        'python single_patient_graph_tester.py'
    )


if __name__ == '__main__':
    main()
