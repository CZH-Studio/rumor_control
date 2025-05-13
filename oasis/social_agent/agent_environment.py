# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
# Licensed under the Apache License, Version 2.0 (the “License”);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an “AS IS” BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from string import Template

from oasis.social_agent.agent_action import SocialAction
from oasis.social_agent.agent_action import ActionType

class Environment(ABC):

    @abstractmethod
    def to_text_prompt(self) -> str:
        r"""Convert the environment to text prompt."""
        raise NotImplementedError


class SocialEnvironment(Environment):
    followers_env_template = Template("I have $num_followers followers.") #Template用于置换字符串中的占位符作为prompt
    follows_env_template = Template("I have $num_follows follows.")

    posts_env_template = Template(
        "After refreshing, you see some posts $posts")
    env_template = Template(
        "$posts_env\npick one you want to perform action that best "
        "reflects your current inclination based on your profile and "
        "posts content. Do not limit your action in just `like` to like posts")

    def __init__(self, action: SocialAction): #初始化可用功能（行动）函数
        self.action = action

    async def get_posts_env(self, post_id) -> str: #刷新获取最新推流消息
        posts = await self.action.refresh(post_id) 
        # TODO: Replace posts json format string to other formats
        if posts["success"]:
            posts_env = json.dumps(posts["posts"], indent=4)
            posts_env = self.posts_env_template.substitute(posts=posts_env)
        else:
            posts_env = "After refreshing, there are no existing posts."
        return posts_env
    
    async def get_posts_list(self) -> list:
        posts = await self.action.refresh(-1) 
        if posts["success"]:
            posts_env = json.dumps(posts["posts"], indent=4)
        else:
            posts_env = []
        return posts_env

    async def get_followers_env(self) -> str:
        # TODO: Implement followers env
        return self.followers_env_template.substitute(num_followers=0)

    async def get_follows_env(self) -> str:
        # TODO: Implement follows env
        return self.follows_env_template.substitute(num_follows=0)
    
    async def get_post(self, post_id: str) -> str:
        message_id = await self.action.channel.write_to_receive_queue( #放入接收队列，等待被DB捕获执行
            (post_id, "", ActionType.FETCH_POST.value))
        post= await self.action.channel.read_from_send_queue(message_id) #从发送队列中读取执行结果
        post = json.dumps(post[2]["posts"][0], indent=4)
        # print("post: ",post)
        return post

    async def to_text_prompt( #覆写父类，将所有刷新的环境信息整合到提示中
        self,
        include_posts: bool = True,
        include_followers: bool = False,
        include_follows: bool = False,
        vaccined: bool = False,
        post_id: int = -1,
    ) -> str:
        followers_env = (await self.get_followers_env()
                         if include_follows else "No followers.")
        follows_env = (await self.get_follows_env()
                       if include_followers else "No follows.")
        if vaccined:
            posts_env = await self.get_posts_env(post_id)
        else:
            posts_env = await self.get_posts_env(post_id) if include_posts else ""
        # print("posts_env: ",posts_env)
        # if vaccined:
        #     message_id = await self.action.channel.write_to_receive_queue( #放入接收队列，等待被DB捕获执行
        #     (post_id, "", ActionType.DRAG_OUT.value))
        #     vacc_posts= await self.action.channel.read_from_send_queue(message_id) #从发送队列中读取执行结果
        #     vacc_posts = json.dumps(vacc_posts[2]["posts"][0], indent=4)
        #     # print("vacc_posts: ",vacc_posts)
        #     index = posts_env.index("you see some posts [\n")
        #     insertion_point = index + len("you see some posts ")
        #     posts_env = posts_env[:insertion_point] + vacc_posts + posts_env[insertion_point:]
        #     # print("posts_env: ",posts_env)
        #     return self.env_template.substitute(
        #         followers_env=followers_env,
        #         follows_env=follows_env,
        #         posts_env=posts_env,
        #     )
        # else:
        #     return self.env_template.substitute(
        #         followers_env=followers_env,
        #         follows_env=follows_env,
        #         posts_env=posts_env,
        #     )
        return self.env_template.substitute(
                followers_env=followers_env,
                follows_env=follows_env,
                posts_env=posts_env,
            )
