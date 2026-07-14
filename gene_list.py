from pathlib import Path
import pandas as pd

def cid_remover(entry):
    entry = int(entry[3:])
    return entry

def main():
    cwd = Path.cwd()
    input_path = Path(cwd,'bio-decagon_folders')

    combo_file = Path(input_path, 'bio-decagon-combo.csv')
    mono_file = Path(input_path, 'bio-decagon-mono.csv')
    target_file = Path(input_path, 'bio-decagon-targets-all.csv')
    ppi_file = Path(input_path, 'bio-decagon-ppi.csv')
    target_file = Path(input_path, 'bio-decagon-targets-all.csv')
    combo = pd.read_csv(combo_file)
    mono = pd.read_csv(mono_file)
    ppi = pd.read_csv(ppi_file)
    target = pd.read_csv(target_file)

    # Add every gene into one combined column
    names = target[(~target['Gene'].isin(ppi['Gene 1'])) & (~target['Gene'].isin(ppi['Gene 2']))]
    print(names)
    gene_df = pd.concat([ppi['Gene 1'], ppi['Gene 2'], names['Gene']]).drop_duplicates().reset_index(drop=True).to_frame(name='Genes')

    gene_df.to_csv(Path(input_path, 'bio-decagon-genes.csv'), index=False)

    # Replace CID for numerical identifiers only
    combo['STITCH 1'] = combo['STITCH 1'].apply(lambda x: cid_remover(x))
    combo['STITCH 2'] = combo['STITCH 2'].apply(lambda x: cid_remover(x))
    mono['STITCH'] = mono['STITCH'].apply(lambda x: cid_remover(x))
    target['STITCH'] = target['STITCH'].apply(lambda x: cid_remover(x))

    combo.to_csv(combo_file, index=False)
    mono.to_csv(mono_file, index=False)
    target.to_csv(target_file, index=False)


