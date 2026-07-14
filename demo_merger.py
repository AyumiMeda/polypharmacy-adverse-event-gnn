import pandas as pd
from pathlib import Path

def outcome_sorter(row):
    if row is None or row.empty:
        return None
    ser_dict = {
    'DE': 1,  # Death
    'LT': 2,  # Life-threatening
    'HO': 3,  # Hospitalization
    'DS': 4,  # Disability
    'CA': 5,  # Congenital anomaly
    'OT': 6   # Other serious
    }
    max_outcome = 'OT'
    for value in row:
        if ser_dict.get(value, 4) < ser_dict.get(max_outcome, 4):
            max_outcome = value
    return max_outcome

def main(csv_folder, cleaner_file):
    cwd = Path.cwd()
    part_year = csv_folder.name[-4:]
    demo = pd.read_csv(csv_folder / f'demo_{part_year}.csv')
    outcome = pd.read_csv(csv_folder / f'outcome_{part_year}.csv')
    reaction = pd.read_csv(csv_folder / f'reaction_{part_year}.csv')

    dt_csv = pd.read_csv(cleaner_file)

    # Rename dt_csv's caseid
    dt_csv = dt_csv.rename(columns={'caseid_x': 'caseid'})
    dt_primaryid_csv = dt_csv['primaryid'].drop_duplicates().reset_index(drop=True)

    # Joining tables

    # Report sources actually not needed

    # 1. outcome -> demo
    outcome = outcome.groupby(['primaryid', 'caseid']).apply(
        lambda x : outcome_sorter(x['outc_cod'])
    ).reset_index(name='most_serious_outcome')

    demo = pd.merge(demo, outcome, on=['primaryid', 'caseid'], how='left')

    # 2. reaction -> demo
    reaction = reaction.groupby(['primaryid', 'caseid']).agg({
        'pt' : lambda x: ','.join(str(i) if pd.notna(i) else 'Uknown' for i in x)
    })
    demo = pd.merge(demo, reaction, on=['primaryid', 'caseid'], how='left')

    # 3. drug/therapy -> demo
    demo = pd.merge(demo, dt_primaryid_csv, on=['primaryid'], how='inner')
    output_file_path = Path(cwd, 'output_csv', f'demo_merger_out')
    output_file_path.mkdir(parents=True, exist_ok=True)
    demo.to_csv(output_file_path / f'entire_demo_{part_year}.csv', index=False)