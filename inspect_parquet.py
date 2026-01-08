
import pandas as pd
import os

base_path = "/Users/simon/Repositories/deleuze2/graphrag_project/output"
entities_path = os.path.join(base_path, "entities.parquet")
relationships_path = os.path.join(base_path, "relationships.parquet")

try:
    if os.path.exists(entities_path):
        df_entities = pd.read_parquet(entities_path)
        print("ENTITIES COLUMNS:", df_entities.columns.tolist())
        print(df_entities.head(3))
    else:
        print(f"File not found: {entities_path}")

    if os.path.exists(relationships_path):
        df_relationships = pd.read_parquet(relationships_path)
        print("RELATIONSHIPS COLUMNS:", df_relationships.columns.tolist())
        print(df_relationships.head(3))
    else:
        print(f"File not found: {relationships_path}")

except Exception as e:
    print(f"Error: {e}")
