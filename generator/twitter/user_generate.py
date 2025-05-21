# import pandas as pd
# import numpy as np
# import random
# import json
# import ast
# from typing import List

# def generate_agent_info(input_json: str, output_csv: str, reference_csv: str = "False_Business_0.csv") -> None:
#     # 1. 从输入JSON中加载数据并采样200个用户
#     with open(input_json, 'r', encoding='utf-8') as f:
#         user_data = json.load(f)
    
#     # 确保有足够样本，不超过总用户数
#     sample_size = min(200, len(user_data))
#     sampled_users = random.sample(user_data, sample_size)
    
#     # 2. 生成基础字段
#     df_output = pd.DataFrame()
#     df_output["user_id"] = range(len(sampled_users))
#     df_output["username"] = [f"user{i}" for i in df_output["user_id"]]
#     df_output["name"] = [f"user_{i}" for i in df_output["user_id"]]
#     df_output["description"] = [user.get("bio", "") for user in sampled_users]  # 使用bio字段
    
#     # 3. 生成following_count和followers_count（基于参考数据分布）
#     df_ref = pd.read_csv(reference_csv)

#     # 过滤无效值并计算对数正态分布参数
#     valid_following = df_ref["following_count"][df_ref["following_count"] > 0]
#     valid_followers = df_ref["followers_count"][df_ref["followers_count"] > 0]

#     # 计算对数空间的参数
#     log_following_mean = np.log(valid_following.mean())
#     log_following_std = np.log(valid_following.std() + 1e-6)  # 避免零标准差

#     log_followers_mean = np.log(valid_followers.mean())
#     log_followers_std = np.log(valid_followers.std() + 1e-6)

#     # 生成符合对数正态分布的值
#     df_output["following_count"] = np.clip(
#         np.random.lognormal(
#             mean=log_following_mean,
#             sigma=log_following_std,
#             size=len(df_output)
#         ),
#         0, 500
#     ).astype(int)

#     df_output["followers_count"] = np.clip(
#         np.random.lognormal(
#             mean=log_followers_mean,
#             sigma=log_followers_std,
#             size=len(df_output)
#         ),
#         0, 1000
#     ).astype(int)
    
#     # 4. 生成activity_level_frequency（从参考数据中采样模式）
#     def sample_activity_pattern():
#         pattern = random.choice(df_ref["activity_level_frequency"].dropna())
#         try:
#             return ast.literal_eval(pattern)
#         except:
#             return [0] * 24  # 默认值
    
#     df_output["activity_level_frequency"] = [str(sample_activity_pattern()) for _ in range(len(df_output))]
    
#     # 5. 生成user_char（使用bio前30字符简化处理）
#     df_output["user_char"] = df_output["description"].str.slice(0, 30) + "..."
    
#     # 6. 生成following_agentid_list（符合2-8定律）
#     num_users = len(df_output)
#     high_profile_num = max(1, int(num_users * 0.2))  # 至少1个高影响力用户
    
#     # 根据followers_count确定高影响力用户
#     high_profile_ids = df_output.sort_values("followers_count", ascending=False).head(high_profile_num).index.tolist()
    
#     def generate_following_list(idx: int, following_count: int) -> List[int]:
#         if following_count <= 0:
#             return []
        
#         # 动态调整概率权重
#         weights = np.ones(num_users)
#         weights[high_profile_ids] = 4  # 高影响力用户被关注的权重更高
#         weights[idx] = 0  # 不能关注自己
        
#         # 确保有足够的候选
#         valid_candidates = [i for i in range(num_users) if i != idx]
#         if not valid_candidates:
#             return []
            
#         if len(valid_candidates) <= following_count:
#             return valid_candidates
        
#         return random.choices(
#             population=valid_candidates,
#             weights=weights[valid_candidates],
#             k=following_count
#         )
    
#     df_output["following_agentid_list"] = [
#         str(generate_following_list(idx, row["following_count"]))
#         for idx, row in df_output.iterrows()
#     ]
    
#     # 7. 添加空字段以匹配参考格式
#     df_output["previous_tweets"] = "[]"
#     df_output["tweets_id"] = "[]"
#     df_output["activity_level"] = df_output["activity_level_frequency"].apply(
#         lambda x: str(["off_line"] * 24)  # 简单模拟
#     )
    
#     # 8. 保存结果
#     df_output.to_csv(output_csv, index=False, quoting=1)  # quoting=1避免列表被错误解析

# # 使用示例
# if __name__ == "__main__":
#     generate_agent_info(
#         input_json="data/my_users_1000_agents.json",  # JSON列表文件，每个元素包含bio字段
#         output_csv="generator/twitter/generated_agents.csv",
#         reference_csv="data/twitter_dataset/anonymous_topic_200_1h/False_Business_0.csv"
#     )
import json
import random
import ast
import pandas as pd
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import ast

import json
import random
import ast
import pandas as pd
import numpy as np

def build_agent_info(
    bios_json_path: str,
    false_business_csv: str,
    output_csv: str,
    sample_size: int = 200,
    popular_ratio: float = 0.2,
    popular_edge_ratio: float = 0.8,
    num_communities: int = 5,
    community_intra_ratio: float = 0.7,
    seed: int = 42,
):
    """
    Sample user bios and expand them into a more decentralized and appropriately sparse agent info network.

    Args:
        bios_json_path: Path to input JSON file of user bios.
        false_business_csv: Path to False_Business_0.csv for reference statistics.
        output_csv: Path to write generated agent_info CSV.
        sample_size: Number of users to sample.
        popular_ratio: Fraction of users designated as popular per community.
        popular_edge_ratio: Fraction of edges pointing to any popular user globally.
        num_communities: Number of communities to partition users into.
        community_intra_ratio: Fraction of edges targeting local community popular hubs.
        seed: Random seed.
    """
    random.seed(seed)
    np.random.seed(seed)

    # Load bios
    with open(bios_json_path, 'r', encoding='utf-8') as f:
        bios = json.load(f)

    sampled = random.sample(bios, k=sample_size)

    # Load reference data
    ref = pd.read_csv(false_business_csv)
    # Parse reference following_agentid_list to get true degree distribution
    ref['following_agentid_list'] = ref['following_agentid_list'].apply(ast.literal_eval)
    degree_dist = ref['following_agentid_list'].apply(len).values
    followers_counts_ref = ref['followers_count'].values
    act_freq_ref = ref['activity_level_frequency'].apply(ast.literal_eval).tolist()

    # Sample following_count from reference degree distribution (enforce sparsity)
    following_counts = np.random.choice(degree_dist, size=sample_size, replace=True)
    # Ensure counts do not exceed possible
    following_counts = np.minimum(following_counts, sample_size - 1)
    

    # Sample followers_count and activity levels
    followers_counts = np.random.choice(followers_counts_ref, size=sample_size, replace=True)
    activity_levels = random.choices(act_freq_ref, k=sample_size)

    # Assign communities
    ids = list(range(sample_size))
    random.shuffle(ids)
    communities = {i: [] for i in range(num_communities)}
    for idx, uid in enumerate(ids):
        communities[idx % num_communities].append(uid)

    # Determine popular hubs per community and globally
    community_popular = {}
    global_popular = set()
    for c, members in communities.items():
        k = max(1, int(len(members) * popular_ratio))
        hubs = set(random.sample(members, k))
        community_popular[c] = hubs
        global_popular.update(hubs)
    non_popular = set(ids) - global_popular

    # Build agent rows
    agent_rows = []
    for i in ids:
        F_i = int(following_counts[i])
        edges = set()
        user_comm = next(c for c, m in communities.items() if i in m)
        local_hubs = community_popular[user_comm]

        tries = 0
        while len(edges) < F_i and tries < F_i * 10:
            r = random.random()
            if r < community_intra_ratio and local_hubs:
                target = random.choice(list(local_hubs))
            else:
                if random.random() < popular_edge_ratio and global_popular:
                    target = random.choice(list(global_popular))
                else:
                    target = random.choice(list(non_popular))
            if target != i:
                edges.add(target)
            tries += 1

        agent_rows.append({
            'id': i,
            'username': f'user{i}',
            'name': f'user_{i}',
            'description': sampled[i]['bio'],
            'following_count': F_i,
            'followers_count': int(followers_counts[i]),
            'user_char': sampled[i].get('user_char', ''),
            'activity_level_frequency': activity_levels[i],
            'following_agentid_list': list(edges),
            'mbti': sampled[i]["mbti"],
            'previous_tweets': '[]',
            'tweets_id': '[]',
            'activity_level': '["off_line"] * 24',
            'created_at': '2021-01-01 00:00:00',
        })

    # Output DataFrame
    out_df = pd.DataFrame(agent_rows)
    out_df['activity_level_frequency'] = out_df['activity_level_frequency'].apply(str)
    out_df['following_agentid_list'] = out_df['following_agentid_list'].apply(str)
    sorted_df = out_df.sort_values(by='id', ascending=True)
    sorted_df.to_csv(output_csv, index=False)


if __name__ == '__main__':
    build_agent_info(
        bios_json_path='data/my_users_1000_agents.json',
        false_business_csv='data/twitter_dataset/anonymous_topic_200_1h/False_Business_0.csv',
        output_csv='generator/twitter/generated_agents.csv',
        sample_size=150
    )
    # visualize_user_network('generator/twitter/generated_agents.csv', max_nodes=100)

