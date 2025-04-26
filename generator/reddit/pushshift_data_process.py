import fastparquet as fp
import ujson
import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from collections import defaultdict
from zhipuai import ZhipuAI

LLM_MODEL = "glm-4-flash"

# Set your OpenAI API key
client = ZhipuAI(api_key="785c33a36f4248c6b76e9b702099aeb1.YP3BG0Re59L9cH2t")

# 读取 Parquet
def read_parquet(file_path):
    df = fp.ParquetFile(file_path).to_pandas()

    # 写入 JSON（ujson 比标准 json 更快）
    with open('data/reddit/data_science_00.json', 'w') as f:
        for record in df.to_dict(orient='records'):
            f.write(ujson.dumps(record) + '\n')  # 逐行写入

def transform_dataset(input_file, output_file, max_items=10000):
    grouped_comments = defaultdict(list)
    item_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if item_count >= max_items:
                break
            try:
                comment = json.loads(line.strip())  # 解析单行JSON
                if comment.get("body") == "[deleted]":
                    continue
                link_id = comment.get("link_id")
                if link_id:
                    grouped_comments[link_id].append(comment)
                item_count += 1
            except json.JSONDecodeError as e:
                print(f"跳过无效行: {line[:50]}... (错误: {e})")
                continue
    
    # 构建目标格式
    transformed_data = []
    for comments in grouped_comments.values():
        post_entry = {"RS": {"title": "None", "selftext": ""}}
        for rc_index, comment in enumerate(comments, 1):
            rc_key = f"RC_{rc_index}"
            group = random.choice(['up', 'control', 'down'])
            post_entry[rc_key] = {
                "body": comment.get("body", ""),
                "group": group
            }
        transformed_data.append(post_entry)
    
    # 保存结果
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(transformed_data, f, indent=2, ensure_ascii=False)
        
def create_selftext(data):
    retry = 5
    while retry > 0:
        try:
            prompt = f""" Please generate a contextual and smooth selftext and title for these comment(RC_1,RC_2,...)
 and notice that the comments are correct: ’{data}’. The
 selftext should be approximately 300 characters long and
 provide relevant information or analysis. The title should 
 be less than 10 characters long and summarize the whole post. 
            Ensure the output can be directly parsed to **JSON**, do not output anything else."""  # noqa: E501

            try:
                response = client.chat.completions.create(
                    model="glm-4",
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    response_format={"type": "json_object"}  # 强制返回JSON格式
                )
                
                # 先打印原始响应检查格式
                raw_response = response.choices[0].message.content
                # print(f"Raw API response: {raw_response}")  # 调试用
                
                # 尝试解析JSON
                profile = json.loads(raw_response)
                return profile
                
            except json.JSONDecodeError as e:
                print(f"Failed to parse JSON: {e}")
                print(f"Raw content: {raw_response}")
                raise
            except Exception as e:
                print(f"API call failed: {e}")
                raise
        except Exception as e:
            print(f"selftext generation failed: {e}. Retrying...")
        retry -= 1


def generate_selftext(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    item_count = len(data)
    print("JSON数据中的项数为:", item_count)
    
    user_data = []
    start_time = datetime.now()
    max_workers = 10  # Adjust according to your system capability
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(create_selftext(data[i])) for i in range(item_count)]
        for i, future in enumerate(as_completed(futures)):
            print(f"Processing {i+1}/{item_count} selftext&title...")
            profile = future.result()
            user_data.append(profile)
            elapsed_time = datetime.now() - start_time
            print(f"Generated {i+1}/{item_count} selftext&title. Time elapsed: "
                  f"{elapsed_time}")
    return user_data

def save_user_data(user_data, filename):
    with open(filename, 'w') as f:
        json.dump(user_data, f, ensure_ascii=False, indent=2)

    
if __name__ == "__main__":
    # read_parquet('data/reddit/science-00000-of-00015-43e0428812705bed.parquet')
    # max_items = 10000
    # transform_dataset('data/reddit/data_science_00.json', 'data/reddit/data_science_00_100.json',max_items)
    user_data = generate_selftext('data/reddit/data_science_00_100.json')
    output_path = 'data/reddit/data_science_00_100_selftext.json'
    save_user_data(user_data, output_path)
    print(f"Generated selftext and saved to {output_path}")