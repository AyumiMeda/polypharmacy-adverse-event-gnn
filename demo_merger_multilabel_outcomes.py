import pandas as pd
from pathlib import Path


OUTCOME_CODES = ['DE', 'LT', 'HO', 'DS', 'CA', 'RI', 'OT']

OUTCOME_RANK = {
    'DE': 1,
    'LT': 2,
    'HO': 3,
    'DS': 4,
    'CA': 5,
    'RI': 6,
    'OT': 7
}


def outcome_sorter(row):
    values = [value for value in row if pd.notna(value) and value in OUTCOME_RANK]
    if not values:
        return None
    return min(values, key=lambda value: OUTCOME_RANK[value])


def main(csv_folder, cleaner_file):
    cwd = Path.cwd()
    part_year = csv_folder.name[-4:]

    demo = pd.read_csv(csv_folder / f'demo_{part_year}.csv')
    outcome = pd.read_csv(csv_folder / f'outcome_{part_year}.csv')
    reaction = pd.read_csv(csv_folder / f'reaction_{part_year}.csv')
    dt_csv = pd.read_csv(cleaner_file)

    dt_csv = dt_csv.rename(columns={'caseid_x': 'caseid'})
    dt_primaryid_csv = dt_csv['primaryid'].drop_duplicates().reset_index(drop=True)

    most_serious = (
        outcome.groupby(['primaryid', 'caseid'])['outc_cod']
        .agg(outcome_sorter)
        .reset_index(name='most_serious_outcome')
    )

    outcome_binary = outcome[['primaryid', 'caseid', 'outc_cod']].copy()
    outcome_binary = outcome_binary[outcome_binary['outc_cod'].isin(OUTCOME_CODES)]
    outcome_binary['value'] = 1

    outcome_multilabel = (
        outcome_binary.pivot_table(
            index=['primaryid', 'caseid'],
            columns='outc_cod',
            values='value',
            aggfunc='max',
            fill_value=0
        )
        .reset_index()
    )

    for code in OUTCOME_CODES:
        if code not in outcome_multilabel.columns:
            outcome_multilabel[code] = 0

    outcome_multilabel = outcome_multilabel[
        ['primaryid', 'caseid'] + OUTCOME_CODES
    ]

    outcome_merged = pd.merge(
        most_serious,
        outcome_multilabel,
        on=['primaryid', 'caseid'],
        how='outer'
    )

    demo = pd.merge(
        demo,
        outcome_merged,
        on=['primaryid', 'caseid'],
        how='left'
    )

    demo['has_outcome_label'] = demo[OUTCOME_CODES].notna().any(axis=1)

    labelled_mask = demo['has_outcome_label']
    demo.loc[labelled_mask, OUTCOME_CODES] = (
        demo.loc[labelled_mask, OUTCOME_CODES]
        .fillna(0)
        .astype(int)
    )

    reaction = reaction.groupby(['primaryid', 'caseid']).agg({
        'pt': lambda x: ','.join(
            str(i) if pd.notna(i) else 'Uknown'
            for i in x
        )
    })

    demo = pd.merge(
        demo,
        reaction,
        on=['primaryid', 'caseid'],
        how='left'
    )

    demo = pd.merge(
        demo,
        dt_primaryid_csv,
        on=['primaryid'],
        how='inner'
    )

    output_file_path = Path(cwd, 'output_csv', 'demo_merger_out')
    output_file_path.mkdir(parents=True, exist_ok=True)

    demo.to_csv(
        output_file_path / f'entire_demo_{part_year}.csv',
        index=False
    )
