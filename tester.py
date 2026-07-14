from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader

cwd = Path.cwd()
emb_map = torch.load(cwd /'output_csv' / 'pre-embeddings' / 'pre_embeddings.pt')
print(emb_map['pt'].keys())