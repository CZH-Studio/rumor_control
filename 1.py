import pandas as pd
import sqlite3
db_path = "False_Business_0.db"
conn = sqlite3.connect(db_path)
query = "SELECT post_id, user_id, original_post_id, content, created_at FROM post"
df = pd.read_sql(query, conn)
# outputpath = 'result_test.csv'
# df.to_csv(outputpath, sep=',', index=False, header=True)
print(df)