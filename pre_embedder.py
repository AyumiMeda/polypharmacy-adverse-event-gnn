import pandas as pd
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
import math
import ast

def norm_text(x):
    #Turn NaN/None into empty string and strip spaces
    if x is None:
        return ""
    if isinstance(x, float) and math.isnan(x):
        return ""
    return str(x).strip()

def embed_text(texts, tokenizer, device, model, batch_size=16) -> torch.Tensor:
    # texts : list[string]

    all_embeds = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = [norm_text(t) for t in texts[i:i + batch_size]]

            # tokenise
            encode = tokenizer(
                batch,
                # pad to the longest in batch to preserve shape
                padding = True,
                # cut off if over the maximum length
                truncation = True,
                max_length=64,
                # Return pytorch tensors
                return_tensors= "pt"
            )

            # move tokens to device
            encode = {k: v.to(device) for k, v in encode.items()}

            # Forward pass -> hidden states for each token (embedding)
            last_hidden = model(**encode).last_hidden_state

            # Pooling tokens -> one vector per entry text
            # Use attention-mask mean pooling to capture contributions from all tokens
            mask = encode['attention_mask'].unsqueeze(-1) # [B,T,1]
            tot_sum = (last_hidden * mask).sum(dim=1) #[B,H]
            counts = mask.sum(dim=1).clamp(min=1) #[B,1]
            pooled = tot_sum/counts

            all_embeds.append(pooled.cpu())

    return torch.cat(all_embeds, dim=0)
def main(drug_file, demo_file, bio_side_effects_file, bio_effect_cat_file, smile_string_file):
    cwd = Path.cwd()
    # Move processing to GPU if possible
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained('dmis-lab/biobert-base-cased-v1.1')
    model = AutoModel.from_pretrained('dmis-lab/biobert-base-cased-v1.1').to(device)
    model.eval()

    # Load csvs
    drug_ther = pd.read_csv(drug_file)
    patient = pd.read_csv(demo_file)
    bio_side_effects = pd.read_csv(bio_side_effects_file)
    bio_effect_cat = pd.read_csv(bio_effect_cat_file)
    smile_string = pd.read_csv(smile_string_file)

    # Convert aggregated values into columns with individual unique entries
    drug_ther['prod_ai'] = drug_ther['prod_ai'].apply(ast.literal_eval)
    chemical_col = pd.Series(drug_ther.explode('prod_ai')['prod_ai'].unique())

    drug_ther['indi_pt'] = drug_ther['indi_pt'].str.split(r',\s*') # I kinda formatted this column weirdly
    indi_pt_col = pd.Series(drug_ther.explode('indi_pt')['indi_pt'].unique())

    patient['pt_2'] = patient['pt_2'].str.split(',')
    pt_col = pd.Series(patient.explode('pt_2')['pt_2'].unique())

    # Columns from tables that need to be embedded
    text_cols = {
    'drugname' : drug_ther['drugname'],
    'chemical' : chemical_col,
    'pt' : pt_col,
    'disease_class' : bio_effect_cat['Disease Class'],
    'side_effect' : bio_side_effects['Side Effect Name'],
    'indi_pt' : indi_pt_col
    }

    # Text embeddings
    for col in text_cols:
        text_cols[col] = text_cols[col].apply(norm_text).astype(str)

    # Unique values for each columns
    unique_per_col = {col: sorted(text_cols[col].unique().tolist()) for col in text_cols}

    # Embed each unique list
    emb_map = {}
    for col in text_cols:
        embs = embed_text(unique_per_col[col], tokenizer, device, model, batch_size=16)
        emb_map[col] = {val: emb for val, emb in zip(unique_per_col[col], embs)}

    # Use CHEM-BERT for chemical structure pre-embedding
    tokenizer = AutoTokenizer.from_pretrained('seyonec/ChemBERTa-zinc-base-v1')
    model = AutoModel.from_pretrained('seyonec/ChemBERTa-zinc-base-v1').to(device)
    model.eval()
    unique_smile = sorted(smile_string['SMILE'].unique().tolist())
    emb = embed_text(unique_smile, tokenizer, device, model, batch_size=16)
    emb_map['SMILE'] = {smile: emb for smile, emb in zip(unique_smile, emb)}

    # Save
    output_path = Path(cwd, 'output_csv', 'pre-embeddings', 'pre_embeddings.pt')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(emb_map, output_path)
    print('Saved embeddings')


