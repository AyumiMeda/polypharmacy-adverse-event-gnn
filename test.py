import ast
from pathlib import Path
import pandas as pd

the_list = []

cluster_csv = Path.cwd() / 'output_csv' / 'demo_clean' / 'cluster_representatives.csv'
df = pd.read_csv(cluster_csv)
file_path = Path.cwd() / 'output_csv' / 'demo_clean' / 'categories.txt'
with open(file_path, 'r') as f:
    content = f.read()
    # Replace '][' with ',' to make it one valid list
    content = content.replace('][', ',')
    # Evaluate the string safely into a Python list
    the_list = ast.literal_eval(content)

unique_list = sorted(list(set(the_list)))
print(len(unique_list))

series = pd.Series(the_list)
df['cluster 1'] = series

fraction = unique_list[0:100]
print(f"I want you to print exactly 100 (to match the number of reactions in prompt) cluster broader names to these adverse reactions. Make them lower case and in a list format (one line):{fraction}")