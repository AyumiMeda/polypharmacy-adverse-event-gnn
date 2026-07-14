import pandas as pd
from pathlib import Path
from datetime import timedelta
from dateutil.relativedelta import relativedelta
import numpy as np

#---------------------------------------------------------
#   Functions
#---------------------------------------------------------

def duration_to_date(duration, unit):
    if pd.isna(duration) or pd.isna(unit):
        return None

    unit = unit.upper()
    try:
        duration = int(duration)
    except:
        duration = 0

    # Absurdity check
    if duration < 0 or duration > 10000:  # example limit, adjust as needed
        return pd.Timedelta(0)

    if unit == 'YR':
        # Approximate a year as 365 days
        return pd.Timedelta(days=365 * duration)
    elif unit == 'MON':
        # Approximate a month as 30 days
        return pd.Timedelta(days=30 * duration)
    elif unit == 'WK':
        return pd.Timedelta(weeks=duration)
    elif unit == 'DAY':
        return pd.Timedelta(days=duration)
    elif unit == 'HR':
        return pd.Timedelta(hours=duration)
    elif unit == 'MIN':
        return pd.Timedelta(minutes=duration)
    elif unit == 'SEC':
        return pd.Timedelta(seconds=duration)
    else:
        # Unknown unit
        return pd.Timedelta(0)

def date_to_duration(start, end, id):
    start = pd.to_datetime(start, errors='coerce')
    end = pd.to_datetime(end, errors='coerce')

    if pd.isna(start) or pd.isna(end):
        return (np.nan, None)

    date_dur = (end - start).days
    rel_date_dur = relativedelta(end, start)
    if date_dur > 365:
        return rel_date_dur.years if rel_date_dur.years > 0 else date_dur // 365, 'YR'
    elif date_dur > 31:
        return rel_date_dur.months if rel_date_dur.months > 0 else date_dur // 30, 'MON'
    elif date_dur > 0:
        return date_dur, 'DAY'
    elif date_dur == 0:
        return 1, 'DAY'
    else:
        return (np.nan, None)

def estimate_date(row, type, median_arg):
    median_durs = median_arg.get(row['drugname'], pd.Timedelta(days=30))
    if median_durs.days <= 0:
        median_durs = pd.Timedelta(days=30)
    if type == 'end':
        row['end_dt'] = row['start_dt'] + median_durs
    elif type == 'start':
        row['start_dt'] = row['end_dt'] - median_durs

    return row


def overlap_dt(group):
    # Sort by dates
    group = group.sort_values('start_dt').copy()
    group['overlap_flag'] = False
    start = group['start_dt'].values
    end = group['end_dt'].values

    n = len(group)

    for i in range(n):
        for j in range(i+1, n):
            if start[j] <= end[i] and start[i] <= end[j]:
                group.at[group.index[i], 'overlap_flag'] = True
                group.at[group.index[j], 'overlap_flag'] = True
            else:
                group.at[group.index[i], 'overlap_flag'] = False
                group.at[group.index[j], 'overlap_flag'] = False

    return group[group['overlap_flag']]

def format_date_or_timedelta(x):
    if pd.isna(x):
        return 'None'
    elif isinstance(x, pd.Timedelta) or isinstance(x, timedelta):
        # Format timedelta as string
        return str(x)
    elif hasattr(x, 'strftime'):
        return x.strftime('%Y-%m-%d')
    else:
        return str(x)

#---------------------------------------------------------
#   Standalone Script
#---------------------------------------------------------
def main(merge_file):
    cwd = Path.cwd()
    part_year = merge_file.name[19:-4]
    input_file_path = merge_file
    dt_csv= pd.read_csv(input_file_path)
    dt_csv['start_dt'] = pd.to_datetime(dt_csv['start_dt'], errors='coerce')
    dt_csv['end_dt'] = pd.to_datetime(dt_csv['end_dt'], errors='coerce')

    #
    # Replace known dates
    #

    print(dt_csv['start_dt'].min(), dt_csv['start_dt'].max())
    print(dt_csv['end_dt'].min(), dt_csv['end_dt'].max())
    valid_dates_mask = (
            (dt_csv['start_dt'] > pd.Timestamp('1950-01-01')) &
            (dt_csv['start_dt'] < pd.Timestamp('2030-01-01')) &
            (dt_csv['end_dt'] > pd.Timestamp('1950-01-01')) &
            (dt_csv['end_dt'] < pd.Timestamp('2030-01-01'))
    )
    dt_csv = dt_csv.loc[valid_dates_mask]

    # Replace the empty end dates
    mask1 = dt_csv['start_dt'].notna() & dt_csv['end_dt'].isna() & dt_csv['dur'].notna() & dt_csv['dur_cod'].notna()
    dt_csv.loc[mask1, 'end_dt'] = dt_csv.loc[mask1].apply(
        lambda x: x['start_dt'] + duration_to_date(x['dur'], x['dur_cod']),
        axis=1
    )

    # Replace the empty start dates
    mask2 = dt_csv['start_dt'].isna() & dt_csv['end_dt'].notna() & dt_csv['dur'].notna() & dt_csv['dur_cod'].notna()
    dt_csv.loc[mask2, 'start_dt'] = dt_csv.loc[mask2].apply(
        lambda x: x['end_dt'] - duration_to_date(x['dur'], x['dur_cod']),
        axis=1
    )

    #
    # Replace dates that are mistakes
    # Some of the dates are invalid like 1975 or go back in time so need to remove those
    #

    mask = (dt_csv['end_dt'] - dt_csv['start_dt']).dt.days.between(0, 6570)
    dt_csv = dt_csv.loc[mask].copy()

    # Create estimates
    valid_dt = dt_csv[dt_csv['start_dt'].notna() & dt_csv['end_dt'].notna()]
    median_dur = valid_dt.groupby('drugname').apply(
        lambda x: np.median(x['end_dt'] - x['start_dt']),
        include_groups=False
    )

    # Replace empty end dates with estimates
    mask3 = dt_csv['end_dt'].isna() & dt_csv['start_dt'].notna()
    dt_csv.loc[mask3, 'end_dt'] = dt_csv.loc[mask3].apply(
        lambda x: estimate_date(x, 'end', median_dur),
        axis=1)

    # Replace empty start dates with estimates
    mask4 = dt_csv['end_dt'].notna() & dt_csv['start_dt'].isna()
    dt_csv.loc[mask4, 'start_dt'] = dt_csv.loc[mask4].apply(
        lambda x: estimate_date(x, 'start', median_dur),
        axis=1)

    # Replace drugs on the same day
    mask5 = dt_csv['end_dt'] == dt_csv['start_dt']
    dt_csv.loc[mask5, 'end_dt'] = dt_csv.loc[mask5, 'end_dt'] + pd.Timedelta(days=2)

    # Add duration where empty
    mask6 = dt_csv['start_dt'].notna() & dt_csv['end_dt'].notna() & dt_csv['dur'].isna()
    dt_csv.loc[mask6, ['dur', 'dur_cod']] = dt_csv.loc[mask6].apply(
        lambda x: pd.Series(date_to_duration(x['start_dt'], x['end_dt'], x['primaryid'])),
        axis=1
    )

    # Make sure dose is numeric
    dt_csv['dose_amt'] = dt_csv['dose_amt'].apply(
        lambda x: x if type(x) is float or type(x) is int or type(x) is str else np.nan
    )
    dt_csv['dose_amt'] = pd.to_numeric(dt_csv['dose_amt'], errors='coerce')

    #
    # Merge rows to get only poly-pharmacy
    #

    # Aggregate by primary and drugname
    grouped = dt_csv.groupby(['primaryid', 'caseid_x', 'drugname']).agg({
        'drug_seq': 'max',
        'role_cod': 'first',
        'prod_ai': 'first',
        'route': 'first',
        'dechal': 'first',
        'rechal': 'first',
        'dose_amt': 'median',
        'dose_unit': 'first',
        'start_dt': 'min',
        'end_dt': 'max',
        'indi_pt': lambda x: ', '.join(str(i) if pd.notna(i) else 'Uknown' for i in set(x))
    })
    grouped = grouped.reset_index()

    # Find overlapping dates for drugs in each primary id
    grouped = grouped.groupby('primaryid').apply(overlap_dt).reset_index(drop=True)
    grouped = grouped.groupby('primaryid').filter(lambda x: len(x) >=2)
    grouped = grouped.drop(columns=['drug_seq', 'overlap_flag'])

    # Further cleaning for missing values
    grouped['dose_unit'] = grouped['dose_unit'].fillna('Unknown')
    grouped['prod_ai'] = grouped['prod_ai'].fillna('Unknown')
    grouped['route'] = grouped['route'].fillna('Unknown')
    # role_cod is left as NaN
    grouped['dechal'] = grouped['dechal'].fillna('U')
    grouped['rechal'] = grouped['rechal'].fillna('U')
    # Use 0 to fill missing dose_amt and a flag
    grouped['dose_amt_missing'] = grouped['dose_amt'].isna().astype(int)
    grouped['dose_amt'] = grouped['dose_amt'].fillna(0)

    output_file_path = Path(cwd, 'output_csv', f'cleaner_out')
    output_file_path.mkdir(parents=True, exist_ok=True)
    grouped.to_csv(output_file_path / f'drug_therapy_cleaned_{part_year}.csv', index=False)


