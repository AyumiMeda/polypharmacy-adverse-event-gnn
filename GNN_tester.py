import torch
from torch import nn
import numpy as np
from torch_geometric.data import HeteroData
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, precision_recall_curve
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.nn import HeteroConv, SAGEConv, NNConv
from pathlib import Path
import pandas as pd
import math
from tqdm import tqdm
from torch_geometric import transforms as T
from torch_geometric.transforms import RandomLinkSplit


class advGNN(nn.Module):
    def __init__(self, hidden_channels,
                 patient_feat_dim, chemical_feat_dim,
                 demo_edge_feat_dim, num_genes,
                 side_effect_emb,
                 rept_emb, mfr_emb, occr_emb, sex_emb):
        super().__init__()

        # Define embeddings
        self.gene_emb = nn.Embedding(num_genes, 128)
        self.side_effect_emb = side_effect_emb
        # Freeze embeddings
        self.side_effect_emb.weight.requires_grad = False
        self.side_effect_proj = nn.Linear(self.side_effect_emb.embedding_dim, 128)
        side_effect_dim = 128

        # Chem projections TEMPORARY
        self.chem_proj = nn.Linear(chemical_feat_dim, 128)

        # Patient projections
        self.patient_proj = nn.Linear(patient_feat_dim, 128)
        self.demo_edge_proj = nn.Linear(demo_edge_feat_dim, 128)

        # Drug projections
        self.drug_proj = nn.Linear(768, 128)

        # Patient embeddings
        self.rept_emb = rept_emb
        self.mfr_emb = mfr_emb
        self.occr_emb = occr_emb
        self.sex_emb = sex_emb

        # Edge networks for NNConv
        chem_edge_network = nn.Sequential(
            nn.Linear(side_effect_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128 * hidden_channels)
        )
        demo_edge_network = nn.Sequential(
            nn.Linear(128, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 128 * 128)
        )
        # HeteroConv layer
        self.conv1 = HeteroConv({
            ('chemical', 'targets', 'chemical'): NNConv(
                in_channels=128,
                out_channels=hidden_channels,
                nn=chem_edge_network,
                aggr='mean'
            ),
            ('chemical', 'rev_targets', 'chemical'): NNConv(
                in_channels=128,
                out_channels=hidden_channels,
                nn=chem_edge_network,
                aggr='mean'
            ),
            ('chemical', 'targets', 'gene'): SAGEConv((-1, -1), hidden_channels),
            ('gene', 'rev_targets', 'chemical'): SAGEConv((-1, -1), hidden_channels),
            ('chemical', 'targets', 'disease_class'): SAGEConv((-1, -1), hidden_channels),
            ('drugs', 'targets', 'chemical'): SAGEConv((-1, -1), hidden_channels),
            ('chemical', 'rev_targets', 'drugs'): SAGEConv((-1, -1), hidden_channels),
            ('drugs', 'targets', 'indi_pt'): SAGEConv((-1, -1), hidden_channels),
            ('indi_pt', 'rev_targets', 'drugs'): SAGEConv((-1, -1), hidden_channels),
            ('patient', 'targets', 'drugs'): NNConv(
                in_channels=128,
                out_channels=hidden_channels,
                nn=demo_edge_network,
                aggr='mean'
            ),
            ('patient', 'targets', 'pt'): SAGEConv((-1, -1), hidden_channels),
            ('drugs', 'rev_targets', 'patient'): SAGEConv((-1, -1), hidden_channels),
            ('gene', 'targets', 'gene'): SAGEConv((-1, -1), hidden_channels),
            ('gene', 'rev_targets', 'gene'): SAGEConv((-1, -1), hidden_channels),
            ('pt', 'targets', 'disease_class'): SAGEConv((-1, -1), hidden_channels),
        }, aggr='mean')

        self.edge_predictor = nn.Sequential(
            nn.Linear(2 * hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1)
        )

    def forward(self, batch):
        device = next(self.parameters()).device

        # Copy x_dict to acoid in memory modification
        x_dict = batch.x_dict.copy()
        edge_index_dict = batch.edge_index_dict
        edge_attr_dict = batch.edge_attr_dict

        # Embed patient node features
        cat_idx = x_dict['patient'][:, :4].long()
        num_feat = x_dict['patient'][:, 4:]
        rept_e = self.rept_emb(cat_idx[:, 0])
        mfr_e = self.mfr_emb(cat_idx[:, 1])
        occr_e = self.occr_emb(cat_idx[:, 2])
        sex_e = self.sex_emb(cat_idx[:, 3])

        # Concatenate embeddings + numerical features
        x_dict['patient'] = torch.cat([rept_e, mfr_e, occr_e, sex_e, num_feat], dim=1)

        # Embed gene nodes and side effect edges
        x_dict['gene'] = self.gene_emb(x_dict['gene'].long()).to(device)
        ids = edge_attr_dict[('chemical', 'targets', 'chemical')].to(device)
        edge_emb = self.side_effect_emb(ids)
        edge_attr_dict[('chemical', 'targets', 'chemical')] = self.side_effect_proj(edge_emb)
        ids = edge_attr_dict[('chemical', 'rev_targets', 'chemical')].to(device)
        rev_edge_emb = self.side_effect_emb(ids)
        edge_attr_dict[('chemical', 'rev_targets', 'chemical')] = self.side_effect_proj(rev_edge_emb)

        # Project patient and chem embeddings to the right dimension
        x_dict['patient'] = self.patient_proj(x_dict['patient'])
        x_dict['chemical'] = self.chem_proj(x_dict['chemical'])
        x_dict['drugs'] = self.drug_proj(x_dict['drugs'])

        # Project patients edge to correct dimension
        edge_attr_dict[('patient', 'targets', 'drugs')] = self.demo_edge_proj(
            edge_attr_dict[('patient', 'targets', 'drugs')]
        )

        # Message passing
        x_dict = self.conv1(x_dict, edge_index_dict, edge_attr_dict)

        # Get the edge indices for the batch
        patient_idx, pt_idx = batch.edge_label_index_dict[('patient', 'targets', 'pt')]
        patient_emb = x_dict['patient'][patient_idx]
        pt_emb = x_dict['pt'][pt_idx]

        # Concatenate all
        edge_input = torch.cat([patient_emb, pt_emb], dim=1)
        scores = self.edge_predictor(edge_input)
        return scores



def norm_text(x):
    #Turn NaN/None into empty string and strip spaces
    if x is None:
        return ""
    if isinstance(x, float) and math.isnan(x):
        return ""
    return str(x).strip()

def create_patient_node(graph_data, new_patient_features, connected_drug_indices,
                        rept_emb, mfr_emb, occr_emb, sex_emb):

    device = next(graph_data.parameters()).device if hasattr(graph_data, "parameters") else "cpu"

    # Construct patient feature vector
    cat_idx = torch.tensor(new_patient_features['cat'], dtype=torch.long)
    num_feat = torch.tensor(new_patient_features['num'], dtype=torch.float)

    # Embed categorical features
    rept_e = rept_emb(cat_idx[0])
    mfr_e = mfr_emb(cat_idx[1])
    occr_e = occr_emb(cat_idx[2])
    sex_e = sex_emb(cat_idx[3])

    patient_x = torch.cat([rept_e, mfr_e, occr_e, sex_e, num_feat], dim=0).unsqueeze(0)

    # Add a new patient node
    new_patient_idx = graph_data['patient'].num_nodes
    graph_data['patient'].x = torch.cat([graph_data['patient'].x, patient_x], dim=0)
    graph_data['patient'].num_nodes += 1

    # Add edges to drugs
    graph_data.edge_index_dict[('patient', 'targets', 'drugs')] = torch.tensor(
        [[new_patient_idx] * len(connected_drug_indices), connected_drug_indices], dtype=torch.long)

    # Dummy edge features
    if ('patient', 'targets', 'drugs') in graph_data.edge_attr_dict:
        num_edges = len(connected_drug_indices)
        new_edge_attr = torch.zeros(num_edges, graph_data['patient', 'targets', 'drugs'].edge_attr.size(1))
        graph_data.edge_attr_dict[('patient', 'targets', 'drugs')] = torch.cat(
            [graph_data.edge_attr_dict[('patient', 'targets', 'drugs')], new_edge_attr], dim=0
        )

    return new_patient_idx, graph_data


# ---------------- Main ----------------
def main():
    cwd = Path.cwd()
    data = torch.load(cwd / 'output_csv' / 'graph_data' / 'graph_data.pt')
    demo = pd.read_csv(cwd / 'output_csv' / 'demo_clean' / 'demo_clean.csv')
    emb_map = torch.load(Path(cwd, 'output_csv', 'pre-embeddings', 'pre_embeddings.pt'))

    # side effect embeddings path
    side_effect_tensor = torch.stack(list(emb_map['side_effect'].values()))
    side_effect_map = {s: i for i, s in enumerate(emb_map['side_effect'].keys())}
    side_effect_emb = nn.Embedding.from_pretrained(side_effect_tensor, freeze=False)

    # patient categorical embeddings
    demo['rept_cod'] = demo['rept_cod'].fillna('UNKNOWN')
    rept_emb = nn.Embedding(len(demo['rept_cod'].unique()), 4)
    valid_sex = {'F', 'M', 'UNK'}
    demo['sex'] = demo['sex'].fillna("UNK")
    demo['sex'] = demo['sex'].apply(lambda x: x if x in valid_sex else 'UNK')
    sex_emb = nn.Embedding(3, 4)
    demo['mfr_sndr'] = demo['mfr_sndr'].fillna('UNKNOWN')
    mfr_sndr_emb = nn.Embedding(len(demo['mfr_sndr'].unique()), 8)
    demo['occr_country'] = demo['occr_country'].fillna('UNKNOWN')
    occr_emb = nn.Embedding(len(demo['occr_country'].unique()), 6)

    # Concatenate for x_dict
    data['patient'].x = torch.cat([
        torch.tensor(data['patient'].cat_index, dtype=torch.long),
        torch.tensor(data['patient'].numerical, dtype=torch.float)
    ], dim=1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = advGNN(
        hidden_channels=128,
        patient_feat_dim=24,
        chemical_feat_dim=data['chemical'].x.size(1),
        demo_edge_feat_dim=data['patient', 'targets', 'drugs'].edge_attr.size(1),
        num_genes=data['gene'].num_nodes,
        side_effect_emb=side_effect_emb,
        rept_emb=rept_emb,
        mfr_emb=mfr_sndr_emb,
        occr_emb=occr_emb,
        sex_emb=sex_emb,
    ).to(device)

    model.load_state_dict(torch.load("gnn_model_epoch10_bce.pth"))
    model.eval()

    # Create new patient node with drug links

    drug_input = Path(cwd, 'output_csv', 'drug_cid', 'drug_cid_attached.csv')
    drug_df = pd.read_csv(drug_input)

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

    drug_df = drug_df.dropna(subset=['drugname'])
    drug_df['drugname'] = drug_df['drugname'].apply(norm_text)
    unique_drug = drug_df.drop_duplicates(subset='drugname')
    drug_link_map = {
        t: i for i, t in enumerate(unique_drug['drugname'])
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

    patient_idx = len(demo['primaryid']) + 1
    # patient node
    rept_idx = rept_map[rept_cod]
    mfr_idx = mfr_sndr_map[mfr_sndr]
    occr_idx = occr_map[occr_country]
    sex_idx = sex_map[gender]

    # Categorical features
    new_cat = torch.tensor([rept_idx, mfr_idx, occr_idx, sex_idx], dtype=torch.long).unsqueeze(0)

    # Numerical features
    new_num = torch.tensor([event_yr, age_years], dtype=torch.float).unsqueeze(0)
    new_patient_features = torch.cat([new_cat, new_num], dim=1)

    # Assign to subgraph
    new_patient_idx = data['patient'].num_nodes
    data['patient'].x = torch.cat([data['patient'].x, new_patient_features], dim=0)
    data['patient'].num_nodes += 1

    patient_to_drug_src = []
    patient_to_drug_dest = []
    patient_to_drug_features = []
    for i in range(drug_count):
        patient_to_drug_src.append(new_patient_idx)
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
        dose_vec = torch.tensor([drug_nodes[i][6]], dtype=torch.float).unsqueeze(0)
        duration_vec = torch.tensor([drug_nodes[i][8]], dtype=torch.float).unsqueeze(0)
        dose_missing_vec = torch.tensor([drug_nodes[i][10]], dtype=torch.float).unsqueeze(0)

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
    patient_to_drug_attr = torch.cat(patient_to_drug_features, dim=0)

    # Add the edges to the graph data
    data.edge_index_dict[('patient', 'targets', 'drugs')] = torch.cat(
        [data.edge_index_dict[('patient', 'targets', 'drugs')],
         patient_to_drug_edge], dim=1
    )

    data.edge_attr_dict[('patient', 'targets', 'drugs')] = torch.cat(
        [data.edge_attr_dict[('patient', 'targets', 'drugs')],
         patient_to_drug_attr], dim=0
    )


    # Create a loader for the new patient node predicting links to all pt nodes
    new_patient_loader = LinkNeighborLoader(
        data,
        num_neighbors={key: [10, 7, 5, 3] for key in data.edge_types},  # same as training
        edge_label_index=(('patient', 'targets', 'pt'),
                          torch.stack([torch.tensor([new_patient_idx] * data['pt'].num_nodes),
                                       torch.arange(data['pt'].num_nodes)], dim=0)), # Only adds the 1 patient with every pt node
        edge_label=torch.zeros(data['pt'].num_nodes),  # dummy, not used
        batch_size=1,  # only 1 patient
        shuffle=False
    )

    model.eval()
    all_probs = []

    for batch in tqdm(new_patient_loader):
        batch = batch.to(device)
        with torch.no_grad():
            logits = model(batch)
            probs = torch.sigmoid(logits).squeeze(-1).cpu().numpy()
            all_probs.append(probs)

    all_probs = np.concatenate(all_probs)
    pt_indices = np.arange(data['pt'].num_nodes)

    # Map pt nodes to probability
    predictions = dict(zip(pt_indices, all_probs))

    # Example threshold
    threshold = 0.78808165

    # Filter pt nodes where predicted probability exceeds threshold
    linked_pt_nodes = [pt for pt, prob in predictions.items() if prob >= threshold]

    # Include probabilities
    linked_pt_nodes_with_probs = [(pt, prob) for pt, prob in predictions.items() if prob >= threshold]

    pt_link_map = {
        p: i for i, p in enumerate(emb_map['pt'].keys())
    }
    idx_to_pt = {i: p for p, i in pt_link_map.items()}

    pt_nodes_list = []
    for pt_nodes in linked_pt_nodes:
        pt_nodes_list.append(idx_to_pt[pt_nodes])
    print("Linked pt nodes idx:", linked_pt_nodes)
    print(f"Linked pt nodes: {pt_nodes_list}" )
    print("Linked pt nodes with probabilities:", linked_pt_nodes_with_probs)




main()
