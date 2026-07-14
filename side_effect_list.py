from pathlib import Path
import pandas as pd

def main():
    cwd = Path.cwd()
    mono_file = Path(cwd, 'bio-decagon_folders', 'bio-decagon-mono.csv')
    combo_file = Path(cwd, 'bio-decagon_folders', 'bio-decagon-combo.csv')

    mono = pd.read_csv(mono_file)
    combo = pd.read_csv(combo_file)
    mono['Side Effect Name'] = mono['Side Effect Name'].str.lower()
    combo['Side Effect Name'] = combo['Side Effect Name'].str.lower()


    side_effect = pd.concat([
        mono['Side Effect Name'],
        combo['Side Effect Name']
    ]).drop_duplicates().reset_index(drop=True).to_frame('Side Effect Name')


    mono.to_csv(mono_file, index=False)
    combo.to_csv(combo_file, index=False)
    output_path = Path(cwd, 'bio-decagon_folders', 'bio-decagon-side_effects.csv')
    side_effect.to_csv(output_path, index=False)