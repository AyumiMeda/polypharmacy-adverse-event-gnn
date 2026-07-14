import pandas as pd
from pathlib import Path
import sqlite3
from itertools import combinations
import gc

def main(folder):
    cwd = Path.cwd()
    for input_file in folder.iterdir():
        stem = Path(input_file).stem
        part = stem.split("_")[-1]
        faers = pd.read_csv(input_file)
        # Connect to database
        conn = sqlite3.connect(str(Path(cwd, 'twosides', 'twosides.db')))
        patient_drugs = faers.groupby('primaryid')['drugname'].apply(list).reset_index()
        pairs =[]
        for _, row in patient_drugs.iterrows():
            primaryid = row['primaryid']
            drugs = row['drugname']

            # Create unique combination pairs
            for drug_1, drug_2 in combinations(drugs, 2):
                pairs.append((primaryid, drug_1, drug_2))

        pairs_df = pd.DataFrame(pairs, columns=['primaryid', 'drug_1', 'drug_2'])

        # Sort drugs alphabetically in each pair so (A,B) and (B,A) are treated the same
        pairs_df['drug_1_sorted'], pairs_df['drug_2_sorted'] = zip(*pairs_df[['drug_1', 'drug_2']].apply(sorted, axis=1))

        # Write FAERS pairs to SQLite
        pairs_df.to_sql('faers_pairs', conn, if_exists='replace', index=False)
        table_name = f"twosides_sorted{part}"

        # Check if twosides_sorted table exists
        exists = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?;", (table_name,)).fetchone()
        if exists:
            print(f"Table '{table_name}' exists!")
        else:
            print(f"Table '{table_name}' does not exist.")

            # Process TWOSIDES in chunks to avoid memory issues
            chunksize = 50_000
            first = True
            for chunk in pd.read_sql_query("SELECT * FROM twosides", conn, chunksize=chunksize):
                chunk['drug_1_sorted'], chunk['drug_2_sorted'] = zip(*chunk[['drug_1_concept_name', 'drug_2_concept_name']].apply(sorted, axis=1))
                chunk.to_sql(f'{table_name}', conn, if_exists='replace' if first else 'append', index=False)
                first = False
        # Create indexes for faster joins
        conn.execute("DROP INDEX IF EXISTS idx_twosides_sorted")
        conn.execute("DROP INDEX IF EXISTS idx_faers_pairs_sorted")
        conn.execute("CREATE INDEX idx_faers_pairs_sorted ON faers_pairs(drug_1_sorted, drug_2_sorted)")
        conn.execute(f"CREATE INDEX idx_twosides_sorted ON {table_name}(drug_1_sorted, drug_2_sorted)")
        conn.commit()

        # Join FAERS pairs with TWOSIDES interactions
        query = f"""
        SELECT f.primaryid, f.drug_1, f.drug_2,
               t.condition_concept_name, 
               t.A, t.B, t.C, t.D, t.PRR,
               t.PRR_error, t.mean_reporting_frequency
        FROM faers_pairs f
        LEFT JOIN {table_name} t
        ON f.drug_1_sorted = t.drug_1_sorted AND f.drug_2_sorted = t.drug_2_sorted
        """
        result = pd.read_sql_query(query, conn)
        conn.execute(f"DROP TABLE IF EXISTS {table_name}")
        conn.close()
        output_path = Path(cwd, 'output_csv', 'twosides_out')
        output_path.mkdir(parents=True, exist_ok=True)
        result.to_csv(Path(output_path, f'{table_name}.csv'), index=False)
        print('finished')
    conn = sqlite3.connect(str(Path(cwd, 'twosides', 'twosides.db')))
    conn.execute('VACUUM;')
    conn.close()


