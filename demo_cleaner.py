import numpy as np
import pandas as pd
from pathlib import Path

def age_coder(age_code):
    if age_code == "A":
        return 41
    elif age_code == "E":
        return 75
    elif age_code == "I":
        return 1
    elif age_code == "N":
        return 0
    elif age_code == "C":
        return 7
    elif age_code == "T":
        return 15

def main():
    cwd = Path.cwd()
    input_file = Path(cwd, 'output_csv', 'db_merger_out', 'combined_demo.csv')
    demo = pd.read_csv(input_file)
    demo['age'] = pd.to_numeric(demo['age'], errors='coerce')
    demo['wt'] = pd.to_numeric(demo['wt'], errors='coerce')

    # Fill in missing age and age cod
    # If the age code is filled, use the average age of the age code
    age_code_mask = demo['age_grp'].notna() & demo['age'].isna()
    demo.loc[age_code_mask, 'age'] = demo.loc[age_code_mask].apply(
        lambda x: age_coder(x['age_grp']), axis = 1
    )
    demo.loc[age_code_mask, 'age_cod'] = 'YR'

    #
    # If their weight is filled in, use the average age of the people in the same weight class with same gender
    #

    # Define bins and labels
    bins = [0, 10, 20, 35, 50, 70, 90, 5000]
    labels = [1, 2, 3, 4, 5, 6, 7]

    # Step 1: male rows with known age
    male_with_age = demo[demo['wt'].notna() & (demo['sex'] == 'M') & (demo['age'].notna())].copy()
    male_with_age['wt'] = pd.to_numeric(male_with_age['wt'], errors='coerce')

    # Step 2: assign bins as integers, not categorical
    male_with_age['wt_bin'] = pd.cut(male_with_age['wt'], bins=bins, labels=labels, right=False).astype(float)

    # Step 3: median age per bin
    wt_median_man = male_with_age.groupby('wt_bin')['age'].median()

    # Step 4: male rows with missing age
    mask_missing = demo['wt'].notna() & (demo['sex'] == 'M') & (demo['age'].isna())
    missing_rows = demo.loc[mask_missing].copy()

    # Step 5: assign bins as integers
    missing_bins = pd.cut(missing_rows['wt'], bins=bins, labels=labels, right=False).astype(int)

    # Step 6: fill age using median
    demo.loc[mask_missing, 'age'] = missing_bins.map(wt_median_man)
    demo.loc[mask_missing, 'age_cod'] = 'YR'

    # Female
    # Step 1: female rows with known age
    female_with_age = demo[demo['wt'].notna() & (demo['sex'] == 'F') & (demo['age'].notna())].copy()
    female_with_age['wt'] = pd.to_numeric(female_with_age['wt'], errors='coerce')

    # Step 2: assign bins as integers, not categorical
    female_with_age['wt_bin'] = pd.cut(female_with_age['wt'], bins=bins, labels=labels, right=False).astype(float)

    # Step 3: median age per bin
    wt_median_woman = female_with_age.groupby('wt_bin')['age'].median()

    # Step 4: male rows with missing age
    femask_missing = demo['wt'].notna() & (demo['sex'] == 'F') & (demo['age'].isna())
    fe_missing_rows = demo.loc[femask_missing].copy()

    # Step 5: assign bins as integers
    fe_missing_bins = pd.cut(fe_missing_rows['wt'], bins=bins, labels=labels, right=False).astype(int)

    # Step 6: fill age using median
    demo.loc[femask_missing, 'age'] = fe_missing_bins.map(wt_median_woman)
    demo.loc[femask_missing, 'age_cod'] = 'YR'

    # Discard weight and weight code and age group column
    demo.drop(columns=['wt', 'wt_cod', 'age_grp'], inplace=True)

    # Replace MON and DAYS with YR


    # Drop age grp column
    output_path = Path(cwd, 'output_csv', 'demo_clean')
    output_path.mkdir(parents=True, exist_ok=True)
    demo.to_csv(Path(output_path, 'demo_clean.csv'), index=False)


