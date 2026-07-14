from pathlib import Path
import pandas as pd
import sqlite3
import numpy as np

#---------------------------------------------------------
#   Functions
#---------------------------------------------------------

def dt_length_chker(entry):
    if pd.notna(entry):
        if isinstance(entry, float) and entry.is_integer():
            entry = str(int(entry))
        else:
            entry = str(entry).strip()
        if len(entry) == 8:
            return pd.to_datetime(entry, format='%Y%m%d', errors='coerce')
        elif len(entry) == 6:
            return pd.to_datetime(entry + '01', format='%Y%m%d', errors='coerce')
        elif len(entry) == 4:
            return pd.to_datetime(entry + '0101', format='%Y%m%d', errors='coerce')
        else:
            return pd.NaT

def change_to_date(column):
    column = column.apply(lambda x: dt_length_chker(x))
    return column

#---------------------------------------------------------
#   Standalone Script
#---------------------------------------------------------
def main(folder):
    # Database file path
    cwd = Path.cwd()
    part_year = folder.name[-4:]
    demo = Path(folder , 'ASCII', f'DEMO{part_year}.txt')
    drug = Path(folder, 'ASCII', f'DRUG{part_year}.txt')
    indication = Path(folder, 'ASCII', f'INDI{part_year}.txt')
    outcome = Path(folder, 'ASCII', f'OUTC{part_year}.txt')
    reaction = Path(folder, 'ASCII', f'REAC{part_year}.txt')
    therapy = Path(folder, 'ASCII', f'THER{part_year}.txt')

    # Extract csv from txt file
    pd.set_option('display.max_columns', None)
    demo_csv = pd.read_csv(demo, delimiter = '$', encoding='latin1')
    drug_csv = pd.read_csv(drug, delimiter = '$', encoding='latin1')
    indication_csv = pd.read_csv(indication, delimiter = '$', encoding='latin1')
    outcome_csv = pd.read_csv(outcome, delimiter = '$', encoding='latin1')
    reaction_csv = pd.read_csv(reaction, delimiter = '$', encoding='latin1')
    therapy_csv = pd.read_csv(therapy, delimiter = '$', encoding='latin1')

    # Names of csv and names
    table_names = [f'demo_{part_year}', f'drug_{part_year}', f'indication_{part_year}',
                   f'outcome_{part_year}', f'reaction_{part_year}', f'therapy_{part_year}']
    table_data = [demo_csv, drug_csv, indication_csv,
                  outcome_csv, reaction_csv, therapy_csv]

    # Clean Demographics Table
    dem_change_dates = ['event_dt', 'mfr_dt', 'init_fda_dt', 'fda_dt', 'rept_dt']
    for col in dem_change_dates:
        demo_csv[col] = change_to_date(demo_csv[col])
    demo_csv.drop(['i_f_code', 'mfr_dt', 'init_fda_dt', 'fda_dt', 'rept_dt'], axis=1, inplace=True)
    demo_csv.drop(['e_sub', 'to_mfr', 'occp_cod', 'reporter_country'], axis=1, inplace=True)
    demo_csv.drop(['lit_ref', 'auth_num', 'mfr_num'], axis=1, inplace=True)
    print('Cleaned demographics')

    # Clean Drugs table
    drug_csv.drop(['val_vbm', 'lot_num', 'exp_dt', 'nda_num'], axis=1, inplace=True)
    print('Cleaned drugs')

    # Clean therapy table
    ther_change_dates = ['start_dt', 'end_dt']
    for col in ther_change_dates:
        therapy_csv[col] = change_to_date(therapy_csv[col])
    therapy_csv = therapy_csv.rename(columns={'dsg_drug_seq': 'drug_seq'})
    print('Cleaned therapy')

    # Clean indications table
    indication_csv = indication_csv.rename(columns={'indi_drug_seq' : 'drug_seq'})

    #
    # Merging
    #

    # Merging therapy and indicator onto drugs
    drug_therapy_csv = pd.merge(drug_csv, indication_csv, on=['primaryid', 'caseid', 'drug_seq'], how='left')
    drug_therapy_csv = pd.merge(drug_therapy_csv, therapy_csv, on=['primaryid', 'drug_seq'], how='left')
    drug_therapy_csv = drug_therapy_csv.dropna(subset=['start_dt', 'end_dt', 'dur'], how='all')
    print(drug_therapy_csv.head())

    # Save csv
    input_file_path = Path(cwd, 'input_csv', f'csv_{part_year}')
    input_file_path.mkdir(parents=True, exist_ok=True)
    for file, name in zip(table_data, table_names):
        file.to_csv(input_file_path / f"{name}.csv", index=False)
    output_file_path = Path(cwd, 'output_csv', f'data_ext_out')
    output_file_path.mkdir(parents=True, exist_ok=True)
    drug_therapy_csv.to_csv(output_file_path / f'drug_therapy_merge_{part_year}.csv', index=False)







