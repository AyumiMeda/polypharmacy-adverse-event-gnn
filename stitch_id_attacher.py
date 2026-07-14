import string
from pathlib import Path
import pandas as pd
import string
import re

def normalise_name(name):
    name = name.lower()
    keep = "-()"
    name = name.translate(str.maketrans(
        '',
        '',
        ''.join(c for c in string.punctuation if c not in keep))
    )
    return name

def split_names(names):
    return [normalise_name(n.strip()) for n in re.split(r'[\\/]', names)]

def take_first(names):
    parts = re.split(r'w/|/|\+| with ', names)
    return parts[0].strip()

def map_to_cids(names : list[str], mapping):
    # Returns all the CIDs for a list of synonyms
    results = []
    for name in names:
        if name in mapping:
            results.append(list(mapping[name])[0])
        else:
            results.append(0)

    return results

def main():
    cwd = Path.cwd()
    file = Path(cwd, 'pubchem', 'CID-Synonym-filtered.gz')
    drug_file = Path(cwd, 'output_csv', 'db_merger_out', 'combined_drug.csv')

    # Replace the prod_ai unknown rows with the drugname
    drug = pd.read_csv(drug_file)
    drug["prod_ai"] = drug["prod_ai"].replace("Unknown", pd.NA)
    drug['drugname'] = drug['drugname'].apply(take_first)
    drug['drugname'] = drug['drugname'].apply(normalise_name)
    drug["prod_ai"] = drug["prod_ai"].fillna(drug["drugname"])

    # Replace the rows to have all the chemicals in drugs in a list
    drug['prod_ai'] = drug['prod_ai'].apply(split_names)
    drug_list = set(n for sublist in drug['prod_ai'] for n in sublist)

    mapping = {}
    print('finished normalising')

    for chunk in pd.read_csv(
        file,
        sep="\t",
        compression="gzip",
        header=None,
        names=["CID", "Synonym"],
        chunksize=1_000_000
    ):
        chunk['synonym_norm'] = chunk['Synonym'].apply(normalise_name)
        filtered = chunk[chunk['synonym_norm'].isin(drug_list)]
        for cid, syn in zip(filtered["CID"], filtered["synonym_norm"]):
            mapping.setdefault(syn, set()).add(cid)

    print('finished mapping')
    drug['CIDs'] = drug['prod_ai'].apply(lambda x: map_to_cids(x, mapping))
    output_path = Path(cwd, 'output_csv', 'drug_cid')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cid_df = pd.DataFrame(list(mapping.items()), columns=["name", "cid"])
    drug.to_csv(Path(output_path,'drug_cid_attached.csv'), index=False)
    cid_df.to_csv(Path(output_path,'cid_synonym.csv'), index=False)




