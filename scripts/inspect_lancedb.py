import lancedb
import pandas as pd

uri = "graphrag_project/output/lancedb"
db = lancedb.connect(uri)
table = db.open_table("default-text_unit-text")

print("Schema:", table.schema)
print("Sample head:")
print(table.to_pandas().head(1))
