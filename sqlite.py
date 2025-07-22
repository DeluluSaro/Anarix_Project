import sqlite3
import pandas as pd
from pathlib import Path


connection  = sqlite3.connect("pr_report.db")

cursor = connection.cursor()


cursor.execute("""
    SELECT name FROM sqlite_master WHERE type='table' AND name='prreport';
""")
table_exists = cursor.fetchone()


if not table_exists:
    cursor.execute("""
        CREATE TABLE prreport(
            pr_id INTEGER,
            rquested_by VARCHAR(25),
            location VARCHAR(25),
            product_name VARCHAR(25),
            product_id INTEGER,
            approved_by VARCHAR(25),
            vendor VARCHAR(25),
            approval_status INTEGER
        )
    """)
    print("Table created!")
else:
    
    cursor.execute('DELETE FROM prreport')
    print("Records are deleted!! (table already existed, so old data cleared)")

data = cursor.execute('''select * from prreport where approval_status = 1''')

for row in data:
    print(row)


csv_files = [
    ("Product-Level Ad Sales and Metrics .csv", "ad_sales_metrics"),
    ("Product-Level Eligibility Table .csv", "eligibility_table"),
    ("Product-Level Total Sales and Metrics .csv", "total_sales_metrics"),
]

data_dir = Path(__file__).parent / "data"

for csv_name, table_name in csv_files:
    csv_path = data_dir / csv_name
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        
        df.to_sql(table_name, connection, if_exists="replace", index=False)
        print(f"Loaded {csv_name} into table {table_name}")
    else:
        print(f"CSV file not found: {csv_path}")

connection.commit()
connection.close()