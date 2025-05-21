# # =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
# # Licensed under the Apache License, Version 2.0 (the “License”);
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #     http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an “AS IS” BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.
# # =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
# import itertools
# import json
# import random
# import time
# from concurrent.futures import ThreadPoolExecutor, as_completed

# import numpy as np
# from rag import generate_user_profile

# total = 1000
# model = "my_users"


# def weighted_random_age(ages, probabilities):
#     ranges = []
#     for age_range in ages:
#         if '+' in age_range:
#             start = int(age_range[:-1])
#             end = start + 20  # Assume 50+ means 50-70
#         else:
#             start, end = map(int, age_range.split('-'))
#         ranges.append((start, end))

#     total_weight = sum(probabilities)
#     rnd = random.uniform(0, total_weight)
#     cumulative_weight = 0
#     for i, weight in enumerate(probabilities):
#         cumulative_weight += weight
#         if rnd < cumulative_weight:
#             start, end = ranges[i]
#             return random.randint(start, end)
#     return None


# def gen_topics():
#     elements = list(range(8))
#     combinations = list(itertools.combinations(elements, 2))
#     expanded_combinations = []
#     while len(expanded_combinations) < total:
#         expanded_combinations.extend(combinations)
#     # Take the first 10,000
#     expanded_combinations = expanded_combinations[:total]
#     # Step 3: Shuffle the order
#     random.shuffle(expanded_combinations)
#     return expanded_combinations


# # Exampls
# ages = ["13-17", "18-24", "25-34", "35-49", "50+"]
# probabilities = [0.066, 0.171, 0.385, 0.207, 0.171]

# professions = [
#     "Agriculture, Food & Natural Resources", "Architecture & Construction",
#     "Arts, Audio/Video Technology & Communications",
#     "Business Management & Administration", "Education & Training", "Finance",
#     "Government & Public Administration", "Health Science",
#     "Hospitality & Tourism", "Human Services", "Information Technology",
#     "Law, Public Safety, Corrections & Security", "Manufacturing", "Marketing",
#     "Science, Technology, Engineering & Mathematics",
#     "Transportation, Distribution & Logistics"
# ]

# topics = [
#     "Politics", "Urban Legends", "Business", "Terrorism & War",
#     "Science & Technology", "Entertainment", "Natural Disasters", "Health",
#     "Education"
# ]

# mbtis = [
#     "ISTJ", "ISFJ", "INFJ", "INTJ", "ISTP", "ISFP", "INFP", "INTP", "ESTP",
#     "ESFP", "ENFP", "ENTP", "ESTJ", "ESFJ", "ENFJ", "ENTJ"
# ]

# genders = ["male", "female", "other"]

# p_mbti = [
#     0.12625, 0.11625, 0.02125, 0.03125, 0.05125, 0.07125, 0.04625, 0.04125,
#     0.04625, 0.06625, 0.07125, 0.03625, 0.10125, 0.11125, 0.03125, 0.03125
# ]
# p_ages = [0.066, 0.171, 0.385, 0.207, 0.171]
# p_genders = [0.4, 0.4, 0.2]
# p_professions = [1 / 16] * 16

# mbti_index = np.random.choice(len(p_mbti), size=total, p=p_mbti)
# age_index = np.random.choice(len(p_ages), size=total, p=p_ages)
# gender_index = np.random.choice(len(p_genders), size=total, p=p_genders)
# profession_index = np.random.choice(len(p_professions),
#                                     size=total,
#                                     p=p_professions)
# topic_index = gen_topics()


# def create_user_profile(i):
#     age = weighted_random_age(ages, probabilities)
#     print(f"Person {i + 1}: Age={age}, MBTI={mbtis[mbti_index[i]]}, Gender="
#           f"{genders[gender_index[i]]}, "
#           f"Profession={professions[profession_index[i]]}")
#     try:
#         return generate_user_profile(age, mbtis[mbti_index[i]],
#                                      genders[gender_index[i]],
#                                      professions[profession_index[i]],
#                                      [topics[x] for x in topic_index[i]])
#     except Exception as e:
#         print(e)
#         retry = 5
#         while retry > 0:
#             try:
#                 return generate_user_profile(
#                     age, mbtis[mbti_index[i]], genders[gender_index[i]],
#                     professions[profession_index[i]],
#                     [topics[x] for x in topic_index[i]])
#             except Exception as e:
#                 print(f"{retry} times", e)
#                 retry -= 1
#         return None


# user_dict = []
# start_time = time.time()

# with ThreadPoolExecutor(max_workers=50) as executor:
#     futures = [executor.submit(create_user_profile, i) for i in range(total)]
#     for future in as_completed(futures):
#         user = future.result()
#         if user:
#             user_dict.append(user)
#         if len(user_dict) % 5000 == 0:
#             print(f"finish {len(user_dict)}")
#             with open(f"./data/{model}_{len(user_dict)}_agents.json",
#                       "w") as f:
#                 json.dump(user_dict, f, indent=4)

# end_time = time.time()
# total_time = end_time - start_time
# print(f"Total time taken: {total_time} seconds")
# print(f"Total users generated: {len(user_dict)}")

# with open(f'./data/{model}_{total}_agents.json', 'w') as f:
#     json.dump(user_dict, f, indent=4)
# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
import itertools
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from zhipuai import ZhipuAI

client = ZhipuAI(api_key="785c33a36f4248c6b76e9b702099aeb1.YP3BG0Re59L9cH2t")

total = 1000
model = "my_users"

# ================== Original gen.py config ==================
ages = ["13-17", "18-24", "25-34", "35-49", "50+"]
probabilities = [0.066, 0.171, 0.385, 0.207, 0.171]

professions = [
    "Agriculture, Food & Natural Resources", "Architecture & Construction",
    "Arts, Audio/Video Technology & Communications",
    "Business Management & Administration", "Education & Training", "Finance",
    "Government & Public Administration", "Health Science",
    "Hospitality & Tourism", "Human Services", "Information Technology",
    "Law, Public Safety, Corrections & Security", "Manufacturing", "Marketing",
    "Science, Technology, Engineering & Mathematics",
    "Transportation, Distribution & Logistics"
]

mbtis = [
    "ISTJ", "ISFJ", "INFJ", "INTJ", "ISTP", "ISFP", "INFP", "INTP", "ESTP",
    "ESFP", "ENFP", "ENTP", "ESTJ", "ESFJ", "ENFJ", "ENTJ"
]

genders = ["male", "female", "other"]

topics = [
    "Politics", "Urban Legends", "Business", "Terrorism & War",
    "Science & Technology", "Entertainment", "Natural Disasters", "Health",
    "Education"
]

p_mbti = [
    0.12625, 0.11625, 0.02125, 0.03125, 0.05125, 0.07125, 0.04625, 0.04125,
    0.04625, 0.06625, 0.07125, 0.03625, 0.10125, 0.11125, 0.03125, 0.03125
]
p_ages = [0.066, 0.171, 0.385, 0.207, 0.171]
p_genders = [0.4, 0.4, 0.2]
p_professions = [1 / 16] * 16

# ================== Modified Generation Logic ==================
def weighted_random_age(ages, probabilities):
    ranges = []
    for age_range in ages:
        if '+' in age_range:
            start = int(age_range[:-1])
            end = start + 20
        else:
            start, end = map(int, age_range.split('-'))
        ranges.append((start, end))

    total_weight = sum(probabilities)
    rnd = random.uniform(0, total_weight)
    cumulative_weight = 0
    for i, weight in enumerate(probabilities):
        cumulative_weight += weight
        if rnd < cumulative_weight:
            start, end = ranges[i]
            return random.randint(start, end)
    return None

def gen_topics():
    elements = list(range(9))
    combinations = list(itertools.combinations(elements, 2))
    expanded_combinations = []
    while len(expanded_combinations) < total:
        expanded_combinations.extend(combinations)
    expanded_combinations = expanded_combinations[:total]
    random.shuffle(expanded_combinations)
    return expanded_combinations

# ================== Integrated User Generation ==================
def generate_user_profile(age, gender, mbti, profession, topics):
    prompt = f"""Please generate a social media user profile based on the provided personal information, including a real name, username, user bio, and a new user persona. The focus should be on creating a fictional background story and detailed interests based on their hobbies and profession.
    Input:
        age: {age}
        gender: {gender}
        mbti: {mbti}
        profession: {profession}
        interested topics: {topics}
    Output:
    {{
        "realname": "str",
        "username": "str",
        "bio": "str",
        "persona": "str"
    }}
    Ensure the output can be directly parsed to **JSON**, do not output anything else."""
    
    try:
        response = client.chat.completions.create(
            model="glm-4-flash",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        profile = json.loads(response.choices[0].message.content)
        profile["topics"] = topics  # Add topics to match original format
        return profile
    except Exception as e:
        print(f"Generation failed: {e}")
        return None

# ================== Main Execution Logic ==================
mbti_index = random.choices(range(len(mbtis)), weights=p_mbti, k=total)
gender_index = random.choices(range(len(genders)), weights=p_genders, k=total)
profession_index = random.choices(range(len(professions)), k=total)
topic_index = gen_topics()

def create_user_profile(i):
    age = weighted_random_age(ages, probabilities)
    gender = genders[gender_index[i]]
    mbti = mbtis[mbti_index[i]]
    profession = professions[profession_index[i]]
    current_topic_indexes = topic_index[i]
    user_topics = [topics[x] for x in current_topic_indexes]
    # topics = [topics[x] for x in topic_index[i]]  # topics需要定义（原代码未完整显示）

    print(f"Person {i+1}: Age={age}, MBTI={mbti}, Gender={gender}, Profession={profession}")

    retry = 5
    while retry > 0:
        try:
            user_profile = generate_user_profile(age, gender, mbti, profession, user_topics)
            if user_profile:
                user_profile.update({
                    "age": age,
                    "gender": gender,
                    "mbti": mbti,
                    "profession": profession,
                    "topics": user_topics  # 确保包含主题信息
                })
                return user_profile
        except Exception as e:
            print(f"Retry {retry}: {e}")
            retry -= 1
    return None

# ================== Execution ==================
if __name__ == "__main__":
    user_dict = []
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=50) as executor:
        futures = [executor.submit(create_user_profile, i) for i in range(total)]
        for future in as_completed(futures):
            user = future.result()
            if user:
                user_dict.append(user)
                if len(user_dict) % 500 == 0:
                    print(f"Generated {len(user_dict)}/{total} profiles")
                    with open(f"./data/{model}_{len(user_dict)}_agents.json", "w") as f:
                        json.dump(user_dict, f, indent=4)

    with open(f'./data/{model}_{total}_agents.json', 'w') as f:
        json.dump(user_dict, f, indent=4)

    print(f"Total time: {time.time()-start_time:.2f}s, Generated: {len(user_dict)}")