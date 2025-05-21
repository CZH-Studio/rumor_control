import pandas as pd
import sqlite3
db_path = "data\simu_db\yaml_gpt\False_Politics_19.db"
conn = sqlite3.connect(db_path)
query = "SELECT post_id, user_id, original_post_id, content, created_at FROM post"
df = pd.read_sql(query, conn)
outputpath = 'result_test.csv'
df.to_csv(outputpath, sep=',', index=False, header=True)
print(df)

# import numpy as np
# import igraph

# # 创建有向图示例
# g = igraph.Graph(directed=True)
# g.add_vertices(5)  # 添加5个顶点
# g.add_edges([(0,1), (1,2), (2,3), (3,4), (4,0), (1,3), (2,0)])  # 添加边
# # 获取入度并转换为NumPy数组
# in_degrees = np.array(g.degree(mode="in"))

# # 获取排序后的索引（从大到小）
# sorted_indices = np.argsort(-in_degrees)  # 负号表示降序排列

# # 输出结果
# print("排序结果 (从大到小):")
# print("顶点ID | 入度")
# print("--------------")
# for idx in sorted_indices:
#     print(f"{idx:6} | {in_degrees[idx]:4}")
    
# import json
# l = [{"id": "1", "name": "John"}, {"id": "2", "name": "Jane"}, {"id": "3", "name": "Doe"}]
# li = json.dumps(l, indent=4)
# print(li)

#    before you output, you MUST check that the ids you selected are in {candidates}.