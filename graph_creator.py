from pathlib import Path
import pandas as pd
import torch
from torch import nn
import ast
import numpy as np
from torch_geometric.data import HeteroData
from tqdm import tqdm
import math
def bidirectional_maker(edge):
    orig_edge = edge
    rev_edge = edge.flip(0)
    bidir_edge = torch.cat([orig_edge, rev_edge], dim=1)
    return bidir_edge

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

    # Edge links between drugs and chemicals
    drug_to_chem_source = []
    drug_to_chem_dest = []
    for _, row in unique_drug.iterrows():
        drug = row['drugname']
        chemicals = row['prod_ai']
        for chemical in chemicals:
            if chemical in chem_link_map:
                drug_to_chem_source.append(drug_link_map[drug])
                drug_to_chem_dest.append(chem_link_map[chemical])
            else:
                print(f"Warning: {chemical} not in chem_link_map")

    drug_to_chem_edge = torch.tensor([drug_to_chem_source,
                                      drug_to_chem_dest],
                                     dtype=torch.long)
    print('finished drug to chem')
    print(len(drug_to_chem_source))

    # Edge links between chemical and gene
    chemical_to_gene_source = []
    chemical_to_gene_dest = []
    chemical_cid_df['cid'] = chemical_cid_df['cid'].apply(extract_first_cid)
    target_df['STITCH'] = target_df['STITCH'].astype(str)
    for chemical in chemical_list:
        # find all genes that the chemical targets
        cid_rows = chemical_cid_df.loc[chemical_cid_df['name'] == chemical, 'cid']
        if cid_rows.empty:
            continue
        chemical_cid = cid_rows.values[0]
        genes = target_df.loc[target_df['STITCH'] == chemical_cid, 'Gene']
        if genes.empty:
            continue
        for gene in genes:
            chemical_to_gene_source.append(chem_link_map[chemical])
            chemical_to_gene_dest.append(gene_link_map[gene])
    chemical_to_gene_edge = torch.tensor([chemical_to_gene_source,
                                          chemical_to_gene_dest],
                                         dtype=torch.long)
    print('finished chem to gene')
    print(len(chemical_to_gene_source))

    # Edge links between demo and drugs
    # map categorical columns to integer IDs
    drug_df['route_id'] = drug_df['route'].map(route_map)
    drug_df['role_id'] = drug_df['role_cod'].map(role_cod_map)
    drug_df['dechal_id'] = drug_df['dechal'].map(chal_map)
    drug_df['rechal_id'] = drug_df['rechal'].map(chal_map)
    drug_df['dose_unit_id'] = drug_df['dose_unit'].map(dose_unit_map)

    # map node indices
    drug_df['demo_idx'] = drug_df['primaryid'].map(demo_link_map)
    drug_df['drug_idx'] = drug_df['drugname'].map(drug_link_map)

    # lookup embedding vectors in batch
    route_vecs = route_emb(torch.tensor(drug_df['route_id'].values))
    role_vecs = role_cod_emb(torch.tensor(drug_df['role_id'].values))
    dechal_vecs = chal_emb(torch.tensor(drug_df['dechal_id'].values))
    rechal_vecs = chal_emb(torch.tensor(drug_df['rechal_id'].values))
    dose_unit_vecs = dose_unit_emb(torch.tensor(drug_df['dose_unit_id'].values))

    # prepare numerical features
    dose_amt_vecs = torch.tensor(drug_df['dose_amt'].fillna(0).values, dtype=torch.float).unsqueeze(1)
    duration_vecs = torch.tensor(drug_df['duration_days'].fillna(0).values, dtype=torch.float).unsqueeze(1)
    dose_amt_missing_vecs = torch.tensor(drug_df['dose_amt_missing'].fillna(0).values, dtype=torch.float).unsqueeze(1)

    # concatenate embeddings
    demo_to_drug_edge_features = torch.cat([
        role_vecs, route_vecs, dechal_vecs, rechal_vecs,
        dose_unit_vecs, dose_amt_vecs, duration_vecs, dose_amt_missing_vecs
    ], dim=1)

    # create edge_index
    demo_to_drug_edge = torch.tensor([
        drug_df['demo_idx'].values,
        drug_df['drug_idx'].values
    ], dtype=torch.long)

    print('finished demo to drug')
    print(len(drug_df['demo_idx'].values))

    # Edge links between genes
    gene_to_gene_src = []
    gene_to_gene_dest = []
    for gene1, gene2 in zip(ppi_df['Gene 1'], ppi_df['Gene 2']):
        gene_to_gene_src.append(gene_link_map[gene1])
        gene_to_gene_dest.append(gene_link_map[gene2])

    gene_to_gene_edge = torch.tensor([gene_to_gene_src,
                                      gene_to_gene_dest],
                                     dtype=torch.long)
    print('finished gene to gene')
    print(len(gene_to_gene_src))

    # Edge links between chemicals
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

    # Build tensors
    src = torch.tensor(merged["idx1"].tolist(), dtype=torch.long)
    dst = torch.tensor(merged["idx2"].tolist(), dtype=torch.long)

    chem_to_chem_edge = torch.stack([src, dst])
    chem_to_chem_edge_features_id = torch.tensor(merged["sid"].tolist(), dtype=torch.long)
    print('finished chemical')
    print(len(src))

    # Edge links between demo and pt
    demo['pt_2'] = demo['pt_2'].str.split(',')
    demo_to_pt_src = []
    demo_to_pt_dest = []
    test_df = demo.explode('pt_2')[['primaryid', 'pt_2']]
    test_df['pt_2'] = test_df['pt_2'].apply(norm_text)
    for _,row in test_df.iterrows():
        demo_to_pt_src.append(demo_link_map[row['primaryid']])
        demo_to_pt_dest.append(pt_link_map[row['pt_2']])
    demo_to_pt_edge = torch.tensor([demo_to_pt_src,
                                    demo_to_pt_dest],
                                   dtype=torch.long)
    print('finished demo to pt')
    print(len(demo_to_pt_src))

    # Edge links between chemicals and disease type
    chem_to_disease_src = []
    chem_to_disease_dest = []
    # Make disease side effect column lower because it has capitalised letters sometimes
    disease_df['Side Effect Name'] = disease_df['Side Effect Name'].str.lower()
    mono_df['STITCH'] = mono_df['STITCH'].astype(str)
    disease_names = set(disease_df['Side Effect Name'])
    for chemical in chemical_list:
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
    print('finished chemical to disease')
    print(len(chem_to_disease_src))

    # Edge links between pt and disease type
    pt_to_disease_src = []
    pt_to_disease_dest = []
    disease_df['Side Effect Name'] = disease_df['Side Effect Name'].str.strip()
    side_effect_names = set(disease_df['Side Effect Name'])
    for pt in pt_link_map.keys():
        pt_mask = pt.lower().strip()
        if pt_mask in side_effect_names:
            disease_cat = disease_df.loc[disease_df['Side Effect Name'] == pt_mask, 'Disease Class'].values[0]
            pt_to_disease_src.append(pt_link_map[pt])
            pt_to_disease_dest.append(disease_link_map[disease_cat])
        else:
            continue
    pt_to_disease_edge = torch.tensor([pt_to_disease_src,
                                       pt_to_disease_dest],
                                      dtype=torch.long)
    print('finished pt to disease')
    print(len(pt_to_disease_src))

    # Edge links between drug and indi_pt
    drug_to_indi_pt_src = []
    drug_to_indi_pt_dest = []
    drug_df['indi_pt'] = drug_df['indi_pt'].str.split(r',\s*')
    indi_pt_dict = (
        drug_df.groupby("drugname")['indi_pt']
        .apply(lambda x: sum(x, []))
        .to_dict()
    )
    for drug in unique_drug['drugname']:
        src_nodes = [drug_link_map[drug]] * len(indi_pt_dict[drug])
        dst_nodes = [indi_pt_link_map[indi_pt] for indi_pt in indi_pt_dict[drug]]

        drug_to_indi_pt_src.extend(src_nodes)
        drug_to_indi_pt_dest.extend(dst_nodes)

    drug_to_indi_edge = torch.tensor(
        [drug_to_indi_pt_src, drug_to_indi_pt_dest],
        dtype=torch.long
    )
    print('finished drug to indi pt')
    print(len(drug_to_indi_pt_src))

    # Node features
    data = HeteroData()

    # chemical nodes
    chemical_features = []
    mono_df['STITCH'] = mono_df['STITCH'].astype(str)
    mono_df['Side Effect Name'] = mono_df['Side Effect Name'].str.lower().apply(norm_text)
    for chem in chemical_list:
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
    data['chemical'].x = torch.stack(chemical_features)
    print('finished chemical features')

    # patient nodes (keep as indexes since features should be trainable)
    # Map categorical columns to indices directly
    rept_idx = demo['rept_cod'].map(rept_map).values
    mfr_idx = demo['mfr_sndr'].map(mfr_sndr_map).values
    occr_idx = demo['occr_country'].map(occr_map).values
    sex_idx = demo['sex'].map(sex_map).values
    # Stack categorical indices into a tensor
    patient_indexes = torch.tensor(
        np.stack([rept_idx, mfr_idx, occr_idx, sex_idx], axis=1),
        dtype=torch.long
    )
    # Numerical features (already vectors, no loop needed)
    patient_num = torch.tensor(
        demo[['event_year', 'age_years']].values,
        dtype=torch.float
    )
    data['patient'].cat_index = patient_indexes
    data['patient'].numerical = patient_num
    print("finished patient features")

    # Drug nodes
    drug_features = []
    for drug in unique_drug['drugname']:
        drugname_idx = drugname_map[drug]
        drugname_vec = drugname_emb(torch.tensor([drugname_idx])).squeeze(0)
        drug_features.append(torch.cat([
            drugname_vec
        ]))
    data['drugs'].x = torch.stack(drug_features)
    print('finished drug features')

    # Indi_pt nodes
    indi_pt_features = []
    for indi_pt in emb_map['indi_pt'].keys():
        indi_pt_idx = indi_pt_map[indi_pt]
        indi_pt_vec = indi_pt_emb(torch.tensor([indi_pt_idx])).squeeze(0)

        indi_pt_features.append(torch.cat([
            indi_pt_vec,
        ]))
    data['indi_pt'].x = torch.stack(indi_pt_features)
    print('finished indi_pt features')

    # Pt nodes
    pt_features = []
    for pt in emb_map['pt'].keys():
        pt_idx = pt_map[pt]
        pt_vec = pt_emb(torch.tensor([pt_idx])).squeeze(0)

        pt_features.append(torch.cat([
            pt_vec,
        ]))
    data['pt'].x = torch.stack(pt_features)

    # Disease nodes
    disease_features = []
    for disease in emb_map['disease_class'].keys():
        disease_idx = disease_map[disease]
        disease_vec = disease_emb(torch.tensor([disease_idx])).squeeze(0)

        disease_features.append(torch.cat([
            disease_vec,
        ]))
    data['disease_class'].x = torch.stack(disease_features)

    # Gene nodes keep as ids (want to train embeddings)
    gene_index = [gene_link_map[gene] for gene in genes_df['Genes']]
    data['gene'].x = torch.tensor(gene_index)

    # Assign edges
    data['chemical', 'targets', 'gene'].edge_index = chemical_to_gene_edge
    data['gene', 'rev_targets', 'chemical'].edge_index = chemical_to_gene_edge.flip(0)
    data['chemical', 'targets', 'chemical'].edge_index = chem_to_chem_edge
    data['chemical', 'targets', 'chemical'].edge_attr = chem_to_chem_edge_features_id
    data['chemical', 'rev_targets', 'chemical'].edge_index = chem_to_chem_edge.flip(0)
    data['chemical', 'rev_targets', 'chemical'].edge_attr = chem_to_chem_edge_features_id
    data['chemical', 'targets', 'disease_class'].edge_index = chem_to_disease_edge
    data['patient', 'targets', 'drugs'].edge_index = demo_to_drug_edge
    data['patient', 'targets', 'drugs'].edge_attr = demo_to_drug_edge_features
    data['drugs', 'rev_targets', 'patient'].edge_index = demo_to_drug_edge.flip(0)
    data['drugs', 'targets', 'indi_pt'].edge_index = drug_to_indi_edge
    data['indi_pt', 'rev_targets', 'drugs'].edge_index = drug_to_indi_edge.flip(0)
    data['drugs', 'targets', 'chemical'].edge_index = drug_to_chem_edge
    data['chemical', 'rev_targets', 'drugs'].edge_index = drug_to_chem_edge.flip(0)
    data['gene', 'targets', 'gene'].edge_index = gene_to_gene_edge
    data['gene', 'rev_targets', 'gene'].edge_index = gene_to_gene_edge.flip(0)
    data['pt', 'targets', 'disease_class'].edge_index = pt_to_disease_edge
    data['patient', 'targets', 'pt'].edge_index = demo_to_pt_edge

    torch.save(data, cwd / "output_csv" / "graph_data" / "graph_data.pt")



main()





