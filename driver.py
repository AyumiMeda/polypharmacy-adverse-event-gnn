import pandas as pd
from pathlib import Path
import data_extract
import drug_therapy_cleaner
import side_effect_list
import stitch_id_attacher

import whole_db_merger
import demo_merger_multilabel_outcomes
import demo_cleaner
import smiles_attacher
import gene_list


# import the files

def main():
    cwd = Path.cwd()
    # Extractor
    #input_file_path = Path(cwd, 'FAERS_folders')
    #for folder in input_file_path.iterdir():
    #    data_extract.main(folder)

    # Drug-therapy Cleaner
    #data_ext_path = Path(cwd, 'output_csv', f'data_ext_out')
    #for file in data_ext_path.iterdir():
    #    drug_therapy_cleaner.main(file)

    # Demographic merger
    #input_csv_folder = Path(cwd, 'input_csv')
    #cleaned_files = Path(cwd, 'output_csv', f'cleaner_out')
    #for folder, file in zip(input_csv_folder.iterdir(), cleaned_files.iterdir()):
    #    demo_merger_multilabel_outcomes.main(folder, file)

    # Quarters merger
    #whole_db_merger.main()

    # Demographic cleaner (done later for better age estimates)
    #demo_cleaner.main()

    # Bio-decagon pubchem ID attacher
    #stitch_id_attacher.main()

    # Bio-decagon gene list
    #gene_list.main()

    # Bio-decagon effects list
    #side_effect_list.main()

    # Pre_embedder
    drug = cwd / 'output_csv' / 'drug_cid' / 'drug_cid_attached.csv'
    demo = cwd / 'output_csv' / 'demo_clean' / 'demo_clean.csv'
    bio_side_effects = cwd / 'bio-decagon_folders' / 'bio-decagon-side_effects.csv'
    bio_effect_cat = cwd / 'bio-decagon_folders' / 'bio-decagon-effectcategories.csv'
    smile_file = cwd / 'output_csv' / 'cid_smile' / 'cid_smile.csv'
    #pre_embedder.main(drug, demo, bio_side_effects, bio_effect_cat, smile_file)

    # GNN creator





main()
