from pathlib import Path
import pandas as pd
import torch
from torch import nn
import ast
import numpy as np
from torch_geometric.data import HeteroData
from tqdm import tqdm
import math


def norm_text(x):
    #Turn NaN/None into empty string and strip spaces
    if x is None:
        return ""
    if isinstance(x, float) and math.isnan(x):
        return ""
    return str(x).strip()

def extract_first_cid(cid_str):
    cid_str = cid_str.strip("{} ")
    first_cid = cid_str.split(",")[0]
    return first_cid.strip()

def main():
    cwd = Path.cwd()

    # Load files
    drug_input = Path(cwd, 'output_csv', 'drug_cid', 'drug_cid_attached.csv')
    demo_input = Path(cwd, 'output_csv', 'demo_clean', 'demo_clean.csv')
    chemical_input = Path(cwd, 'output_csv', 'drug_cid','cid_synonym.csv')
    gene_input = Path(cwd, 'bio-decagon_folders', 'bio-decagon-genes.csv')
    target_file = Path(cwd, 'bio-decagon_folders', 'bio-decagon-targets-all.csv')
    ppi_file = Path(cwd, 'bio-decagon_folders', 'bio-decagon-ppi.csv')
    combo_file = Path(cwd, 'bio-decagon_folders', 'bio-decagon-combo.csv')
    mono_file = Path(cwd, 'bio-decagon_folders', 'bio-decagon-mono.csv')
    disease_file = Path(cwd, 'bio-decagon_folders', 'bio-decagon-effectcategories.csv')
    smile_string_file = Path(cwd, 'output_csv', 'cid_smile', 'cid_smile.csv')
    drug_df = pd.read_csv(drug_input)
    demo = pd.read_csv(demo_input)
    # chemical_cid_df contains the chemical to cid mapping
    chemical_cid_df = pd.read_csv(chemical_input)
    # genes_df contains the list of every gene in the bio-decagon database
    genes_df = pd.read_csv(gene_input)
    # target_df contains what gene each chemical targets
    target_df = pd.read_csv(target_file)
    # ppi_df contains gene to gene interactions
    ppi_df = pd.read_csv(ppi_file)
    # combo_df contains the chemical to chemical side effects
    combo_df = pd.read_csv(combo_file)
    # mono_df contains the mono-chemical side effects
    mono_df = pd.read_csv(mono_file)
    # disease_df contains the side effect categories
    disease_df = pd.read_csv(disease_file)
    # smile_df contains the SMILE string for each cid
    smile_df = pd.read_csv(smile_string_file)


    # Prepare for embedding
    emb_map = torch.load(Path(cwd, 'output_csv', 'pre-embeddings', 'pre_embeddings.pt'))

    # drugname
    drugname_tensor = torch.stack(list(emb_map['drugname'].values()))
    drugname_map = {d:i for i,d in enumerate(emb_map['drugname'].keys())}
    drugname_emb = nn.Embedding.from_pretrained(drugname_tensor, freeze=False)

    # prod_ai
    prod_ai_tensor = torch.stack(list(emb_map['chemical'].values()))
    prod_ai_map = {p:i for i,p in enumerate(emb_map['chemical'].keys())}
    prod_ai_emb = nn.Embedding.from_pretrained(prod_ai_tensor, freeze=False)

    # pt
    pt_tensor = torch.stack(list(emb_map['pt'].values()))
    pt_map = {p:i for i,p in enumerate(emb_map['pt'].keys())}
    pt_emb = nn.Embedding.from_pretrained(pt_tensor, freeze=False)

    # disease class
    disease_tensor = torch.stack(list(emb_map['disease_class'].values()))
    disease_map = {d:i for i,d in enumerate(emb_map['disease_class'].keys())}
    disease_emb = nn.Embedding.from_pretrained(disease_tensor, freeze=False)

    # indi_pt
    indi_pt_tensor = torch.stack(list(emb_map['indi_pt'].values()))
    indi_pt_map = {ip: i for i,ip in enumerate(emb_map['indi_pt'].keys())}
    indi_pt_emb = nn.Embedding.from_pretrained(indi_pt_tensor, freeze=False)

    # side_effects
    side_effect_tensor = torch.stack(list(emb_map['side_effect'].values()))
    side_effect_map = {s: i for i,s in enumerate(emb_map['side_effect'].keys())}
    side_effect_emb = nn.Embedding.from_pretrained(side_effect_tensor, freeze=False)

    # chemical fingerprint
    smile_tensor = torch.stack(list(emb_map['SMILE'].values()))
    smile_map = {s: i for i,s in enumerate(emb_map['SMILE'].keys())}
    smile_emb = nn.Embedding.from_pretrained(smile_tensor, freeze=False)

    # route
    drug_df['route'] = drug_df['route'].str.lower()
    route_map = {r: i for i,r in enumerate(drug_df['route'].unique())}
    route_emb = nn.Embedding(len(route_map), 8)

    # role_cod
    role_cod_map = {'PS': 0, 'SS': 1, 'C':2, 'I':3, 'DN': 4}
    role_cod_emb = nn.Embedding(len(role_cod_map), 4)

    # dechal / rechal
    chal_map = {'Y': 0, 'N': 1, 'U': 2, 'D': 3}
    chal_emb = nn.Embedding(len(chal_map), 4)

    # dose_unit
    drug_df['dose_unit'] = drug_df['dose_unit'].fillna('UNKNOWN')
    dose_unit_map = {d:i for i,d in enumerate(drug_df['dose_unit'].unique())}
    dose_unit_emb = nn.Embedding(len(dose_unit_map), 6)

    # duration
    drug_df['start_dt'] = pd.to_datetime(drug_df['start_dt'])
    drug_df['end_dt'] = pd.to_datetime(drug_df['end_dt'])
    drug_df['duration_days'] = (drug_df['end_dt'] - drug_df['start_dt']).dt.days

    #rept_cod
    demo['rept_cod'] = demo['rept_cod'].fillna('UNKNOWN')
    rept_map = {r:i for i,r in enumerate(demo['rept_cod'].unique())}


    # sex
    valid_sex = {'F', 'M', 'UNK'}
    demo['sex'] = demo['sex'].fillna("UNK")
    demo['sex'] = demo['sex'].apply(lambda x: x if x in valid_sex else 'UNK')
    sex_map = {'F': 0, 'M': 1, 'UNK': 2}


    # age REMOVE AND PLACE IN THE DEMO CLEANER LATER
    unit_to_years = {
        'DY': 1 / 365,
        'HR': 1 / (365 * 24),
        'WK': 1 / 52,
        'MON': 1 / 12,
        'YR': 1,
        'DEC': 10
    }
    demo['age'] = demo['age'].fillna(-1)
    demo['age_cod'] = demo['age_cod'].fillna('YR')
    demo['age_years'] = demo['age'] * demo['age_cod'].map(unit_to_years).fillna(1)

    # mfr_sndr
    demo['mfr_sndr'] = demo['mfr_sndr'].fillna('UNKNOWN')
    mfr_sndr_map = {m:i for i,m in enumerate(demo['mfr_sndr'].unique())}


    # event_dt
    demo['event_dt'] = pd.to_datetime(demo['event_dt'], errors='coerce')
    demo['event_year'] = demo['event_dt'].dt.year
    demo['event_year'] = demo['event_year'].fillna(0).astype(int)

    # occr_country
    demo['occr_country'] = demo['occr_country'].fillna('UNKNOWN')
    occr_map = {o:i for i,o in enumerate(demo['occr_country'].unique())}


    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    # Edge mapper
    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    drug_df = drug_df.dropna(subset=['drugname'])
    drug_df['drugname'] = drug_df['drugname'].apply(norm_text)
    unique_drug = drug_df.drop_duplicates(subset='drugname')
    drug_link_map = {
        t: i for i, t in enumerate(unique_drug['drugname'])
    }

    demo_link_map = {
        t: i for i, t in enumerate(demo['primaryid'])
    }

    drug_df['prod_ai'] = drug_df['prod_ai'].apply(ast.literal_eval)
    drug_df['CIDs'] = drug_df['CIDs'].apply(ast.literal_eval)
    unique_drug['prod_ai'] = unique_drug['prod_ai'].apply(ast.literal_eval)
    unique_drug['CIDs'] = unique_drug['CIDs'].apply(ast.literal_eval)
    # chemical list contains every unique chemical in the drug database
    chemical_list = drug_df.explode('prod_ai')['prod_ai'].unique()
    chem_link_map = {
        c: i for i, c in enumerate(chemical_list)
    }

    gene_link_map = {
        g: i for i, g in enumerate(genes_df['Genes'])
    }

    indi_pt_link_map = {
        ip: i for i, ip in enumerate(emb_map['indi_pt'].keys())
    }

    pt_link_map = {
        p: i for i, p in enumerate(emb_map['pt'].keys())
    }

    disease_link_map = {
        d: i for i, d in enumerate(emb_map['disease_class'].keys())
    }

    # Patient node
    patient = '15Q1_100128784'
    event_yr = 2010
    rept_cod = 'EXP'
    mfr_sndr = 'ABBVIE'
    age_years = 61
    gender = 'F'
    occr_country = 'FR'

    # Drugs nodes
    drug_count = 3
    drug_nodes = [['azathioprine', 'C', 'azathioprine', 'unknown', 'U',
                   'D', 150, 'MG', 770, ["Crohn's disease"], 0], ['humira',
                   'PS', 'adalimumab', 'subcutaneous', 'U', 'D', 100, 'MG',
                   700, ["Crohn's disease"], 0], ['solupred', 'C', 'prednisolone',
                   'unknown', 'U', 'D', 45, 'MG', 15, ['Hepatitis'], 0]]



    # Edge links

    # patient to drugs
    patient_idx = len(demo['primaryid']) + 1
    patient_to_drug_src = []
    patient_to_drug_dest = []
    patient_to_drug_features = []
    for i in range(drug_count):
        patient_to_drug_src.append(patient_idx)
        patient_to_drug_dest.append(drug_link_map[drug_nodes[i][0]])

        # features
        route_id = route_map[drug_nodes[i][3]]
        role_id = role_cod_map[drug_nodes[i][1]]
        dechal_id = chal_map[drug_nodes[i][4]]
        rechal_id = chal_map[drug_nodes[i][5]]
        dose_unit_id = dose_unit_map[drug_nodes[i][7]]

        route_vec = route_emb(torch.tensor([route_id]))
        role_vec = role_cod_emb(torch.tensor([role_id]))
        dechal_vec = chal_emb(torch.tensor([dechal_id]))
        rechal_vec = chal_emb(torch.tensor([rechal_id]))
        dose_unit_vec = dose_unit_emb(torch.tensor([dose_unit_id]))

        # numerical features
        dose_vec = torch.tensor(drug_nodes[i][6], dtype=torch.float).unsqueeze(0).unsqueeze(0)
        duration_vec = torch.tensor(drug_nodes[i][8], dtype=torch.float).unsqueeze(0).unsqueeze(0)
        dose_missing_vec = torch.tensor(drug_nodes[i][10], dtype=torch.float).unsqueeze(0).unsqueeze(0)

        edge_feat = torch.cat([
            route_vec,
            role_vec,
            dechal_vec,
            rechal_vec,
            dose_unit_vec,
            dose_vec,
            duration_vec,
            dose_missing_vec
        ], dim=-1)

        patient_to_drug_features.append(edge_feat)

    patient_to_drug_edge = torch.tensor([patient_to_drug_src,
                                         patient_to_drug_dest],
                                        dtype=torch.long)
    patient_to_drug_attr = torch.stack(patient_to_drug_features, dim=0)

    # drug to indi_pt
    drug_to_indi_src = []
    drug_to_indi_dest = []
    for i in range(drug_count):
        drug_idx = drug_link_map[drug_nodes[i][0]]
        for indi_pt in drug_nodes[i][9]:
            indi_pt_idx = indi_pt_link_map[indi_pt]
            drug_to_indi_src.append(drug_idx)
            drug_to_indi_dest.append(indi_pt_idx)

    drug_to_indi_edge = torch.tensor([drug_to_indi_src,
                                      drug_to_indi_dest],
                                     dtype=torch.long)

    # drug to chemical
    drug_to_chem_src = []
    drug_to_chem_dest = []
    for i in range(drug_count):
        drug_idx = drug_link_map[drug_nodes[i][0]]
        prod_ai_idx = chem_link_map[drug_nodes[i][2]]
        drug_to_chem_src.append(drug_idx)
        drug_to_chem_dest.append(prod_ai_idx)

    drug_to_chem_edge = torch.tensor([drug_to_chem_src,
                                      drug_to_chem_dest],
                                     dtype=torch.long)

    # chemical to gene
    chem_to_gene_src = []
    chem_to_gene_dest = []
    chemical_cid_df['cid'] = chemical_cid_df['cid'].apply(extract_first_cid)
    target_df['STITCH'] = target_df['STITCH'].astype(str)
    genes_target = []
    for i in range(drug_count):
        chemical = drug_nodes[i][2]

        # find all genes that the chemical targets
        cid_rows = chemical_cid_df.loc[chemical_cid_df['name'] == chemical, 'cid']
        if cid_rows.empty:
            continue
        chemical_cid = cid_rows.values[0]
        genes = target_df.loc[target_df['STITCH'] == chemical_cid, 'Gene']
        if genes.empty:
            continue
        for gene in genes:
            genes_target.append(gene)
            chem_to_gene_src.append(chem_link_map[chemical])
            chem_to_gene_dest.append(gene_link_map[gene])
    chemical_to_gene_edge = torch.tensor([chem_to_gene_src,
                                          chem_to_gene_dest],
                                         dtype=torch.long)

    # gene to gene
    gene_to_gene_src = []
    gene_to_gene_dest = []
    for gene in genes_target:
        gene2_values = ppi_df.loc[ppi_df['Gene 1'] == gene, 'Gene 2']
        if gene2_values.empty:
            continue
        else:
            for gene2 in gene2_values:
                gene_to_gene_src.append(gene_link_map[gene])
                gene_to_gene_dest.append(gene_link_map[gene2])

    gene_to_gene_edge = torch.tensor([gene_to_gene_src,
                                      gene_to_gene_dest],
                                     dtype=torch.long)

    # chem to chem
    chem_to_chem_src = []
    chem_to_chem_dest = []
    chem_to_chem_features = []

    combo_df['STITCH 1'] = combo_df['STITCH 1'].astype(str)
    combo_df['STITCH 2'] = combo_df['STITCH 2'].astype(str)
    chemical_cid_df['cid'] = chemical_cid_df['cid'].astype(str)
    combo_df['Side Effect Name'] = combo_df['Side Effect Name'].str.lower().apply(norm_text)

    # keep chemical with cids
    valid_cids = chemical_cid_df.loc[
        chemical_cid_df['name'].isin(chemical_list), 'cid'
    ].unique()

    # filter combos so both STITCH cids are in list
    filtered_combos = combo_df[
        combo_df['STITCH 1'].isin(valid_cids) & combo_df['STITCH 2'].isin(valid_cids)
    ]

    # Map cid to canonical name and drop duplicates
    cid_to_name = chemical_cid_df.drop_duplicates('cid')[['cid', 'name']]

    # Join stitch 1 cid to name
    merged = (
        filtered_combos
        .merge(cid_to_name, left_on='STITCH 1', right_on='cid', how = "inner")
        .rename(columns={'name': 'chem1'})
    )

    # Join STITCH 2 cid to name
    merged = (
        merged
        .merge(cid_to_name, left_on='STITCH 2', right_on='cid', how = "inner")
        .rename(columns={'name': 'chem2'})
    )

    # keep the rows where both names are in the chemical list
    merged = merged[
        merged["chem1"].isin(chemical_list) & merged["chem2"].isin(chemical_list)
        ]

    # Add index to embedding column
    merged['idx1'] = merged['chem1'].map(chem_link_map)
    merged['idx2'] = merged['chem2'].map(chem_link_map)

    # Map side effects to ids
    merged = merged[merged["Side Effect Name"].isin(side_effect_map)]
    merged["sid"] = merged["Side Effect Name"].map(side_effect_map)

    for i in range(drug_count):
        chemical = drug_nodes[i][2]
        chemical_rows = merged.loc[merged['chem1'] == chemical, ['chem2', 'sid']]
        if chemical_rows.empty:
            continue
        for _, row in chemical_rows.iterrows():
            chem2 = row['chem2']
            side_effect_id = row['sid']
            chem_to_chem_src.append(chem_link_map[chemical])
            chem_to_chem_dest.append(chem_link_map[chem2])
            chem_to_chem_features.append(side_effect_id)
    chem_to_chem_edge = torch.tensor([chem_to_chem_src,
                                      chem_to_chem_dest],
                                     dtype=torch.long)

    chem_to_chem_attr = torch.tensor([chem_to_chem_features])

    # chem to disease class
    chem_to_disease_src = []
    chem_to_disease_dest = []
    # Make disease side effect column lower because it has capitalised letters sometimes
    disease_df['Side Effect Name'] = disease_df['Side Effect Name'].str.lower()
    mono_df['STITCH'] = mono_df['STITCH'].astype(str)
    disease_names = set(disease_df['Side Effect Name'])
    for i in range(drug_count):
        chemical = drug_nodes[i][2]
        chemical_cids = chemical_cid_df.loc[chemical_cid_df['name'] == chemical, 'cid']
        if chemical_cids.empty:
            continue
        cid = chemical_cids.values[0]
        side_effects = mono_df.loc[mono_df['STITCH'] == cid, 'Side Effect Name']
        if side_effects.empty:
            continue
        for side_effect in side_effects:
            if side_effect in disease_names:
                disease_cat = disease_df.loc[
                disease_df['Side Effect Name'] == side_effect, 'Disease Class'
                ].values[0]
                chem_to_disease_src.append(chem_link_map[chemical])
                chem_to_disease_dest.append(disease_link_map[disease_cat])
    chem_to_disease_edge = torch.tensor([chem_to_disease_src,
                                         chem_to_disease_dest],
                                        dtype=torch.long)



    # Nodes
    subgraph = HeteroData()

    # patient node
    rept_idx = rept_map[rept_cod]
    mfr_idx = mfr_sndr_map[mfr_sndr]
    occr_idx = occr_map[occr_country]
    sex_idx = sex_map[gender]

    # Categorical features
    patient_indexes = torch.tensor([rept_idx, mfr_idx, occr_idx, sex_idx], dtype=torch.long).unsqueeze(0)  # [1,4]

    # Numerical features
    patient_num = torch.tensor([event_yr, age_years], dtype=torch.float).unsqueeze(0)  # [1,2]

    # Assign to subgraph
    subgraph['patient'].cat_index = patient_indexes
    subgraph['patient'].numerical = patient_num

    # drug node
    drug_features =[]
    for i in range(drug_count):
        drug = drug_nodes[i][0]
        drugname_idx = drugname_map[drug]
        drugname_vec = drugname_emb(torch.tensor([drugname_idx])).squeeze(0)
        drug_features.append(torch.cat([
            drugname_vec
        ]))
    subgraph['drugs'].x = torch.stack(drug_features)

    # chemical node
    chemical_features = []
    mono_df['STITCH'] = mono_df['STITCH'].astype(str)
    mono_df['Side Effect Name'] = mono_df['Side Effect Name'].str.lower().apply(norm_text)

    for i in range(drug_count):
        chem = drug_nodes[i][2]
        name_idx = prod_ai_map[chem]
        name_emb = prod_ai_emb(torch.tensor([name_idx])).squeeze(0)
        chem_cids = chemical_cid_df.loc[chemical_cid_df['name'] == chem, 'cid']
        if chem_cids.empty:
            fingerprint_emb = torch.zeros(768)
            mono_effects_emb = torch.zeros(768)
        else:
            chem_cid = chem_cids.values[0]
            smiles = smile_df.loc[smile_df['CID'] ==chem_cid, 'SMILE']
            if smiles.empty:
                fingerprint_emb = torch.zeros(768)
            else:
                smile = smiles.values[0]
                fingerprint_idx = smile_map[smile]
                fingerprint_emb = smile_emb(torch.tensor([fingerprint_idx])).squeeze(0)
            mono_effects = mono_df.loc[mono_df['STITCH'] ==chem_cid, 'Side Effect Name'].tolist()
            indices = torch.tensor([side_effect_map[mon] for mon in mono_effects], dtype=torch.long)
            mono_effects_emb = side_effect_emb(indices).mean(dim=0)

        chemical_features.append(torch.cat([
            name_emb, fingerprint_emb, mono_effects_emb], dim=0))
    subgraph['chemical'].x = torch.stack(chemical_features)

    # Gene nodes keep as ids
    gene_index = [gene_link_map[gene] for gene in genes_df['Genes']]
    subgraph['gene'].x = torch.tensor(gene_index)

    # Pt nodes
    pt_features = []
    for pt in emb_map['pt'].keys():
        pt_idx = pt_map[pt]
        pt_vec = pt_emb(torch.tensor([pt_idx])).squeeze(0)

        pt_features.append(torch.cat([
            pt_vec,
        ]))
    subgraph['pt'].x = torch.stack(pt_features)

    # Disease nodes
    disease_features = []
    for disease in emb_map['disease_class'].keys():
        disease_idx = disease_map[disease]
        disease_vec = disease_emb(torch.tensor([disease_idx])).squeeze(0)

        disease_features.append(torch.cat([
            disease_vec,
        ]))
    subgraph['disease_class'].x = torch.stack(disease_features)

    # indi_pt nodes
    indi_pt_features = []
    for i in range(drug_count):
        indi_pt_list = drug_nodes[i][9]
        for indi_pt in indi_pt_list:
            indi_idx = indi_pt_map[indi_pt]
            indi_pt_vec = indi_pt_emb(torch.tensor([indi_idx])).squeeze(0)

            indi_pt_features.append(torch.cat([
                indi_pt_vec,
            ]))
    subgraph['indi_pt'].x = torch.stack(indi_pt_features)

    # Assign edges
    subgraph['chemical', 'targets', 'gene'].edge_index = chemical_to_gene_edge
    subgraph['gene', 'rev_targets', 'chemical'].edge_index = chemical_to_gene_edge.flip(0)
    subgraph['chemical', 'targets', 'chemical'].edge_index = chem_to_chem_edge
    subgraph['chemical', 'targets', 'chemical'].edge_attr = chem_to_chem_attr
    subgraph['chemical', 'rev_targets', 'chemical'].edge_index = chem_to_chem_edge.flip(0)
    subgraph['chemical', 'rev_targets', 'chemical'].edge_attr = chem_to_chem_attr
    subgraph['chemical', 'targets', 'disease_class'].edge_index = chem_to_disease_edge
    subgraph['patient', 'targets', 'drugs'].edge_index = patient_to_drug_edge
    subgraph['patient', 'targets', 'drugs'].edge_attr = patient_to_drug_attr
    subgraph['drugs', 'rev_targets', 'patient'].edge_index = patient_to_drug_edge.flip(0)
    subgraph['drugs', 'targets', 'indi_pt'].edge_index = drug_to_indi_edge
    subgraph['indi_pt', 'rev_targets', 'drugs'].edge_index = drug_to_indi_edge.flip(0)
    subgraph['drugs', 'targets', 'chemical'].edge_index = drug_to_chem_edge
    subgraph['chemical', 'rev_targets', 'drugs'].edge_index = drug_to_chem_edge.flip(0)
    subgraph['gene', 'targets', 'gene'].edge_index = gene_to_gene_edge
    subgraph['gene', 'rev_targets', 'gene'].edge_index = gene_to_gene_edge.flip(0)
    subgraph['patient', 'targets', 'pt'].edge_index = torch.empty((2,0), dtype=torch.long)

    torch.save(subgraph, cwd / "output_csv" / "graph_data" / "subgraph_data.pt")

main()




















