import pandas as pd
from pathlib import Path
import ast

def main():
    cwd = Path.cwd()
    input_smile = Path(cwd, 'pubchem', 'CID-SMILES.gz')
    input_drug = Path(cwd, 'output_csv', 'drug_cid', 'drug_cid_attached.csv')
    drug = pd.read_csv(input_drug)
    drug['CIDs'] = drug['CIDs'].apply(ast.literal_eval)
    cid_list = drug.explode('CIDs')['CIDs'].unique().tolist()
    print('Finished list')

    chunksize = 1000000
    smile_dict = {}
    for chunk in pd.read_csv(
            input_smile,
            chunksize=chunksize,
            sep="\t",
            compression="gzip",
            header=None,
            names=["CID", "SMILE"]
    ):
        filtered = chunk[chunk['CID'].isin(cid_list)].reset_index(drop=True)
        for cid, smile in zip(filtered['CID'], filtered['SMILE']):
            smile_dict[cid] = smile

    print('Finished dictionary')
    smile_df = pd.DataFrame(list(smile_dict.items()), columns=["CID", "SMILE"])
    output_path = Path(cwd, 'output_csv', 'cid_smile')
    output_path.mkdir(parents=True, exist_ok=True)
    smile_df.to_csv(Path(output_path,'cid_smile.csv'), index=False)

