from pathlib import Path
import ast
import pandas as pd
import math

def mapper(row):
    items = [x.strip() for x in row.split(",")]
    mapped = [pt_to_cluster[i] for i in items if i in pt_to_cluster]
    # remove duplicates
    return ",".join(sorted(set(mapped)))

cluster_file = Path.cwd() / 'output_csv' / 'demo_clean' / 'cluster_representatives.csv'
demo_clean = Path.cwd() / 'output_csv' / 'demo_clean' / 'demo_clean.csv'
cluster_df = pd.read_csv(cluster_file)
demo_df = pd.read_csv(demo_clean)

pt_to_cluster = dict(zip(cluster_df["pt"], cluster_df["cluster 1"]))

demo_df['pt_2'] = demo_df['pt'].apply(mapper)

demo_df.to_csv(demo_clean, index=False)