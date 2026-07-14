import sqlite3
import pandas as pd

input_file = "C:/Users/minty/Downloads/GitMasterFolder/Adverse_effects/output_csv/cleaner_out/drug_therapy_cleaned_15Q1.csv"
test = pd.read_csv(input_file)

print(test['dose_unit'].unique().tolist())