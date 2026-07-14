from pathlib import Path
import pandas as pd

def main():
    # Assign release batch to primary ids in all databases

    # Attach quarter to primary_id of drug_therapy tables
    cwd = Path.cwd()
    cleaner_folder = Path(cwd, 'output_csv', 'cleaner_out')
    db_merger_folder_1 = Path(cwd, 'output_csv', 'db_merge_prep_dt')
    db_merger_folder_1.mkdir(parents=True, exist_ok=True)

    for file in cleaner_folder.iterdir():
        df = pd.read_csv(file)
        quarter = file.stem[-4:]
        df['primaryid'] = df['primaryid'].apply(lambda x: f'{quarter}_{x}')
        df.to_csv(Path(db_merger_folder_1, f'd_t_attached_{quarter}.csv'), index=False)


    # Attach quarter to primary_id of demo tables
    demo_folder = Path(cwd, 'output_csv', 'demo_merger_out')
    db_merger_folder_2 = Path(cwd, 'output_csv', 'db_merge_prep_demo')
    db_merger_folder_2.mkdir(parents=True, exist_ok=True)
    for file in demo_folder.iterdir():
        df = pd.read_csv(file)
        quarter = file.stem[-4:]
        df['primaryid'] = df['primaryid'].apply(lambda x: f'{quarter}_{x}')
        df.to_csv(Path(db_merger_folder_2, f'demo_attached_{quarter}.csv'), index=False)

    # Quarters merger for drug_therapy
    db_merger_out = Path(cwd, 'output_csv', 'db_merger_out')
    db_merger_out.mkdir(parents=True, exist_ok=True)
    dfs = [pd.read_csv(file) for file in db_merger_folder_1.glob("*.csv")]
    combined = pd.concat(dfs, ignore_index=True)
    combined['route'] = combined['route'].str.lower()
    combined.to_csv(db_merger_out / 'combined_drug.csv', index=False)

    # Quarters merger for demo
    dfs = [pd.read_csv(file) for file in db_merger_folder_2.glob("*.csv")]
    combined = pd.concat(dfs, ignore_index=True)
    combined.to_csv(db_merger_out / 'combined_demo.csv', index=False)