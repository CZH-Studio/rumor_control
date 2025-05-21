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

import inspect
import json
import logging
import sys
from datetime import datetime
from typing import TYPE_CHECKING, Any

from camel.configs import ChatGPTConfig
from camel.memories import (ChatHistoryMemory, MemoryRecord,
                            ScoreBasedContextCreator)
from camel.messages import BaseMessage
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType, OpenAIBackendRole
from camel.utils import OpenAITokenCounter

from oasis.social_agent.agent_action import SocialAction
from oasis.social_agent.agent_environment import SocialEnvironment
from oasis.social_platform import Channel
from oasis.social_platform.config import UserInfo
from mist.data import get_data, get_prompt

from camel.configs import ZHIPUAI_API_PARAMS

if TYPE_CHECKING:
    from oasis.social_agent import AgentGraph

if "sphinx" not in sys.modules:
    agent_log = logging.getLogger(name="social.agent")
    agent_log.setLevel("DEBUG")
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_handler = logging.FileHandler(f"./log/social.agent-{str(now)}.log")
    file_handler.setLevel("DEBUG")
    file_handler.setFormatter(
        logging.Formatter(
            "%(levelname)s - %(asctime)s - %(name)s - %(message)s"))
    agent_log.addHandler(file_handler)


class SocialAgent:
    r"""Social Agent."""

    def __init__(
        self,
        agent_id: int,
        user_info: UserInfo,
        twitter_channel: Channel,
        inference_channel: Channel = None,
        model_type: str = "llama-3",
        agent_graph: "AgentGraph" = None,
        action_space_prompt: str = None,
        is_openai_model: bool = False,
        mist_type: str = "MIST-20",
    ):
        self.agent_id = agent_id
        self.user_info = user_info
        self.twitter_channel = twitter_channel
        self.infe_channel = inference_channel
        self.action_space_prompt = action_space_prompt
        self.env = SocialEnvironment(SocialAction(agent_id, twitter_channel)) #环境初始化
        self.system_message = BaseMessage.make_assistant_message( #继承自camel的系统消息类
            role_name="User",
            content=self.user_info.to_system_message(action_space_prompt), #构建用户行动提示词
        )
        # print("system message: ",self.system_message.content)
        self.model_type = model_type #默认llama-3
        self.is_openai_model = is_openai_model #默认不是openai模型
        self.mist_type = mist_type #默认MIST-20
        #openai模型建立
        model_config = ChatGPTConfig( #通用config
                tools=self.env.action.get_openai_function_list(), #提供模型可用的功能函数
                temperature=0.5,
                # tool_choice="required",
            )
        if self.is_openai_model:
            self.model_backend = ModelFactory.create( #创建openai模型
                model_platform=ModelPlatformType.OPENAI, #基于openai平台
                model_type=ModelType(self.model_type), #openai模型类型
                model_config_dict=model_config.as_dict(), #注入参数
            )
        else:
            model_config_dict = model_config.as_dict()
            filtered_config = {
                k: v for k, v in model_config_dict.items()
                if k in ZHIPUAI_API_PARAMS
            }
            self.model_backend = ModelFactory.create( #创建glm模型
                model_platform=ModelPlatformType.ZHIPU,
                model_type=ModelType(self.model_type),
                model_config_dict=filtered_config,
            )
       

        context_creator = ScoreBasedContextCreator( #定义openai格式的上下文创建器，最大token为4096
            # OpenAITokenCounter(ModelType.GPT_3_5_TURBO),
            OpenAITokenCounter(ModelType(model_type)),
            4096,
        )
        self.memory = ChatHistoryMemory(context_creator, window_size=5)
        self.system_message = BaseMessage.make_system_message( #构建系统消息
            role_name="system",
            content=self.user_info.to_system_message(
                action_space_prompt),  # system prompt
        )
        self.agent_graph = agent_graph
        self.test_prompt = (
            "\n"
            "Helen is a successful writer who usually writes popular western "
            "novels. Now, she has an idea for a new novel that could really "
            "make a big impact. If it works out, it could greatly "
            "improve her career. But if it fails, she will have spent "
            "a lot of time and effort for nothing.\n"
            "\n"
            "What do you think Helen should do?")
        news, self.mist_label = get_data(self.mist_type)
        self.mist_test_prompt = get_prompt(self.user_info.get_user_description(), news)
        self.private_message_id = -1

    async def perform_action_by_llm(self):
        if self.private_message_id != -1:
            env_prompt = await self.env.to_text_prompt(vaccined=True, post_id=self.private_message_id)
            self.private_message_id = -1
        else:
            env_prompt = await self.env.to_text_prompt() #将所有刷新的环境信息整合到提示中
        user_prompt = self.user_info.get_user_description() #将用户信息整合到提示中
        user_msg = BaseMessage.make_user_message(
            role_name="User",
            # content=(
            #     f"Please perform social media actions after observing the "
            #     f"platform environments. Notice that don't limit your "
            #     f"actions for example to just like the posts. "
            #     f"Here is your social media environment: {env_prompt}"),
            content=(
                f"Your actions should be consistent with your self-description and personality:{user_prompt}.Please perform social media actions like a human after observing the "
                f"platform environments. {self.action_space_prompt}"
                f"Here is your social media environment: {env_prompt}"
                ),
        )
        self.memory.write_record(
            MemoryRecord(
                message=user_msg,
                role_at_backend=OpenAIBackendRole.USER,
            ))

        openai_messages, _ = self.memory.get_context()
        content = ""
        # sometimes self.memory.get_context() would lose system prompt
        # start_message = openai_messages[0]
        # if start_message["role"] != self.system_message.role_name:
        #     # print("role: ",start_message["role"])
        #     openai_messages = [{
        #         "role": self.system_message.role_name,
        #         "content": self.system_message.content,
        #         # "tools": self.env.action.get_openai_function_list(),
        #     }] + openai_messages
        
        # print("openai_messages: ",openai_messages)
        # if not openai_messages: #结构化输出有误
        #     print("openai_messages is empty")
        #     openai_messages = [{
        #         "role": self.system_message.role_name,
        #         "content": self.system_message.content,
        #     }] + [user_msg.to_openai_user_message()]
        agent_log.info(
            f"Agent {self.agent_id} is running with prompt: {openai_messages}")

        # if self.is_openai_model:
        if True:
            try:
                response = self.model_backend.run(openai_messages)
                agent_log.info(f"Agent {self.agent_id} response: {response}")
                content = response
                for tool_call in response.choices[0].message.tool_calls: #support multiple tool calls
                    print("--------------------------")
                    action_name = tool_call.function.name
                    print("action_name:",action_name)
                    args = json.loads(tool_call.function.arguments)
                    print("args:",args)
                    print(f"Agent {self.agent_id} is performing "
                          f"action: {action_name} with args: {args}")
                    result = await getattr(self.env.action, action_name)(**args)
                    agent_log.info(f"Agent {self.agent_id}: {result}")
                    self.perform_agent_graph_action(action_name, args) #执行涉及图更新的action
            except Exception as e:
                print(e)
                agent_log.error(f"Agent {self.agent_id} error: {e}")
                content = "No response."

        # else:
        #     retry = 5
        #     exec_functions = []

        #     while retry > 0:
        #         start_message = openai_messages[0]
        #         if start_message["role"] != self.system_message.role_name:
        #             openai_messages = [{
        #                 "role": self.system_message.role_name,
        #                 "content": self.system_message.content,
        #             }] + openai_messages
        #         mes_id = await self.infe_channel.write_to_receive_queue(
        #             openai_messages)
        #         mes_id, content = await self.infe_channel.read_from_send_queue(
        #             mes_id)

        #         agent_log.info(
        #             f"Agent {self.agent_id} receive response: {content}")

        #         try:
        #             content_json = json.loads(content)
        #             functions = content_json["functions"]
        #             # reason = content_json["reason"]

        #             for function in functions:
        #                 name = function["name"]
        #                 # arguments = function['arguments']
        #                 if name != "do_nothing":
        #                     arguments = function["arguments"]
        #                 else:
        #                     # The success rate of do_nothing is very low
        #                     # It often drops the argument, causing retries
        #                     # It's a waste of time, manually compensating here
        #                     arguments = {}
        #                 exec_functions.append({
        #                     "name": name,
        #                     "arguments": arguments
        #                 })
        #                 self.perform_agent_graph_action(name, arguments)
        #             break
        #         except Exception as e:
        #             agent_log.error(f"Agent {self.agent_id} error: {e}")
        #             exec_functions = []
        #             retry -= 1
        #     for function in exec_functions:
        #         try:
        #             await getattr(self.env.action,
        #                           function["name"])(**function["arguments"])
        #         except Exception as e:
        #             agent_log.error(f"Agent {self.agent_id} error: {e}")
        #             retry -= 1

        #     if retry == 0:
        #         content = "No response."
        agent_msg = BaseMessage.make_assistant_message(role_name="Assistant",
                                                       content=content)
        self.memory.write_record(
            MemoryRecord(message=agent_msg,
                         role_at_backend=OpenAIBackendRole.ASSISTANT))
        
        
    async def perform_vaccine(self, post_id):
            # Get posts:
            env_prompt = await self.env.to_text_prompt(vaccined=True,post_id=post_id) #将所有刷新的环境信息整合到提示中
            user_prompt = self.user_info.get_user_description() #将用户信息整合到提示中
            user_msg = BaseMessage.make_user_message(
                role_name="User",
                content=(
                    f"Your actions should be consistent with your self-description and personality:{user_prompt}.Please perform social media actions like a human after observing the "
                    f"platform environments. {self.action_space_prompt}"
                    f"Here is your social media environment: {env_prompt}"
                    ),
            )
            self.memory.write_record(
                MemoryRecord(
                    message=user_msg,
                    role_at_backend=OpenAIBackendRole.USER,
                ))

            openai_messages, _ = self.memory.get_context()
            agent_log.info(
                f"Agent {self.agent_id} is running with prompt: {openai_messages}")

            # if self.is_openai_model:
            if True:
                try:
                    response = self.model_backend.run(openai_messages)
                    agent_log.info(f"Agent {self.agent_id} response: {response}")
                    content = response
                    for tool_call in response.choices[0].message.tool_calls: #support multiple tool calls
                        print("--------------------------")
                        action_name = tool_call.function.name
                        print("action_name:",action_name)
                        args = json.loads(tool_call.function.arguments)
                        print("args:",args)
                        print(f"Agent {self.agent_id} is performing "
                            f"action: {action_name} with args: {args}")
                        result = await getattr(self.env.action, action_name)(**args)
                        agent_log.info(f"Agent {self.agent_id}: {result}")
                        self.perform_agent_graph_action(action_name, args) #执行涉及图更新的action
                except Exception as e:
                    print(e)
                    agent_log.error(f"Agent {self.agent_id} error: {e}")
                    content = "No response."
            agent_msg = BaseMessage.make_assistant_message(role_name="Assistant",
                                                        content=content)
            self.memory.write_record(
                MemoryRecord(message=agent_msg,
                            role_at_backend=OpenAIBackendRole.ASSISTANT))
    
    async def generate_vaccine(self, post_id):
            source_post = await self.env.get_post(post_id)
            # print("source_post: ", source_post)
            user_msg = BaseMessage.make_user_message(
                role_name="User",
                content=(
                    f"""you see a post: {source_post}. Categorize the topic of the post from the following options:
                     1.Business
                     2.Politics
                     3.Culture & Society
                     4.entertainment
                     Ensure your output is domain, not number, don't output anything else.
                    """
                    ),
            )
            openai_messages = [user_msg.to_openai_user_message()]
            if True:
                try:
                    response = self.model_backend.run(openai_messages)
                    print("domain: ",response.choices[0].message.content)
                    # agent_log.info(f"Agent {self.agent_id} response: {response}")
                    ans = response.choices[0].message.content
                except Exception as e:
                    print(e)
                    ans = "news"
            user_msg = BaseMessage.make_user_message(
                role_name="User",
                content=(
                    f"""you are a knowledgeable expert of the {ans} domain. Now there is a rumer post 
                    on the social media that is factually wrong: {source_post}. Please write a post to inoculate people
                    from the misinformation. Your post content should follow the following format but more concise:
                    1. State the truth first: Lead with the fact if it’s clear, pithy, and sticky—make 
                    it simple, concrete,and plausible. Provide a factual alternative that fills a causal “gap”, 
                    explaining what happened if the misinformation is corrected. Do not rely on a simple 
                    retraction (“this claim is not true”).
                    2. Point to misinformation: Warn that a myth is coming. Repeat the misinformation, 
                    only once, directly prior to the correction.
                    3. Explain why misinformation is wrong: Explain how the myth misleads. Point out 
                    logical or argumentative fallacies underlying the misinformation.
                    4. State the truth again: Finish by reinforcing the fact. Repeat the fact multiple 
                    times if possible.
                    
                    here is an example of a anti-rumer post about climate change that follows the format:
                    Scientists observe human fingerprints all over our climate
The warming effect from greenhouse gases like 
carbon dioxide has been confirmed by many lines 
of evidence. Aircraft and satellites measure less 
heat escaping to space at the exact wavelengths 
that carbon dioxide absorbs energy. The upper 
atmosphere cools while the lower atmosphere 
warms—a distinct pattern of greenhouse 
warming. 
A common climate myth is that climate has always 
changed naturally in the past, therefore modern 
climate change must be natural also.
This argument commits the single cause fallacy, 
falsely assuming that because natural factors 
have caused climate change in the past, then 
they must always be the cause of climate change.
This logic is the same as seeing a murdered body 
and concluding that people have died of natural 
causes in the past, so the murder victim must 
have also died of natural causes.
Just as a detective finds clues in a crime scene, 
scientists have found many clues in climate 
measurements confirming humans are causing 
global warming. Human-caused global warming is 
a measured fact.
                    
                    use a simple, natural and novel toungue so that everyone understand. Don't make it paragraphic.
                    Output: Title: title of the post.\nContent: your content
                    """
                    ),
            )
            openai_messages = [user_msg.to_openai_user_message()]
            if True:
                try:
                    response = self.model_backend.run(openai_messages)
                    # agent_log.info(f"Agent {self.agent_id} response: {response}")
                    vacc_post = response.choices[0].message.content
                    # print("vacc_post: ",vacc_post)
                    return vacc_post
                except Exception as e:
                    print(e)
                    return e
                    
            


    async def perform_mist_test(self):
        # Get posts:
        user_msg = BaseMessage.make_user_message(
            role_name="User",
            content=self.mist_test_prompt,
        )
        openai_messages, _ = self.memory.get_context()
        openai_messages = [user_msg.to_openai_user_message()] + openai_messages
        agent_log.info(
            f"Agent {self.agent_id} is running with prompt: {openai_messages}")

        if True:
            try:
                response = self.model_backend.run(openai_messages)
                agent_log.info(f"Agent {self.agent_id} response: {response}")
                ans = json.loads(response.choices[0].message.content)
            except Exception as e:
                print(e)
                agent_log.error(f"Agent {self.agent_id} error: {e}")
                content = "No response."
        #mist v,r,f,d,n计算
        Veracity, real, fake, distrust, naivite = 0,0,0,0,0
        for i,label in enumerate(self.mist_label):
            if ans[f"news_{i+1}"] == label:
                Veracity += 1
                real += 1 if label == "Real" else 0
                fake += 1 if label == "Fake" else 0
            distrust += 1 if ans[f"news_{i+1}"] == "Fake" else 0
            naivite += 1 if ans[f"news_{i+1}"] == "Real" else 0
        if self.mist_type == "MIST-20":
            distrust = distrust-10 if distrust-10>0 else 0
            naivite = naivite-10 if naivite-10>0 else 0
        return (self.agent_id, Veracity, real, fake, distrust, naivite)
                


    async def perform_test(self):
        """
        doing test for all agents.
        """
        # user conduct test to agent
        _ = BaseMessage.make_user_message(role_name="User",
                                          content=("You are a twitter user."))
        # TODO error occurs
        # self.memory.write_record(MemoryRecord(user_msg,
        #                                       OpenAIBackendRole.USER))

        openai_messages, num_tokens = self.memory.get_context()

        openai_messages = ([{
            "role":
            self.system_message.role_name,
            "content":
            self.system_message.content.split("# RESPONSE FORMAT")[0],
        }] + openai_messages + [{
            "role": "user",
            "content": self.test_prompt
        }])
        agent_log.info(f"Agent {self.agent_id}: {openai_messages}")

        message_id = await self.infe_channel.write_to_receive_queue(
            openai_messages)
        message_id, content = await self.infe_channel.read_from_send_queue(
            message_id)
        agent_log.info(f"Agent {self.agent_id} receive response: {content}")
        return {
            "user_id": self.agent_id,
            "prompt": openai_messages,
            "content": content
        }

    async def perform_action_by_hci(self) -> Any:
        print("Please choose one function to perform:")
        function_list = self.env.action.get_openai_function_list()
        for i in range(len(function_list)):
            agent_log.info(f"Agent {self.agent_id} function: "
                           f"{function_list[i].func.__name__}")

        selection = int(input("Enter your choice: "))
        if not 0 <= selection < len(function_list):
            agent_log.error(f"Agent {self.agent_id} invalid input.")
            return
        func = function_list[selection].func

        params = inspect.signature(func).parameters
        args = []
        for param in params.values():
            while True:
                try:
                    value = input(f"Enter value for {param.name}: ")
                    args.append(value)
                    break
                except ValueError:
                    agent_log.error("Invalid input, please enter an integer.")

        result = await func(*args)
        return result

    async def perform_action_by_data(self, func_name, *args, **kwargs) -> Any: #执行指定动作
        function_list = self.env.action.get_openai_function_list()
        for i in range(len(function_list)):
            if function_list[i].func.__name__ == func_name:
                func = function_list[i].func
                result = await func(*args, **kwargs)
                agent_log.info(f"Agent {self.agent_id}: {result}")
                return result
        raise ValueError(f"Function {func_name} not found in the list.")

    def perform_agent_graph_action(
        self,
        action_name: str,
        arguments: dict[str, Any],
    ):
        r"""Remove edge if action is unfollow or add edge
        if action is follow to the agent graph.
        """
        if "unfollow" in action_name:
            followee_id: int | None = arguments.get("followee_id", None)
            if followee_id is None:
                return
            self.agent_graph.remove_edge(self.agent_id, followee_id)
            agent_log.info(f"Agent {self.agent_id} unfollowed {followee_id}")
        elif "follow" in action_name:
            followee_id: int | None = arguments.get("followee_id", None)
            if followee_id is None:
                return
            self.agent_graph.add_edge(self.agent_id, followee_id)
            agent_log.info(f"Agent {self.agent_id} followed {followee_id}")

    def __str__(self) -> str:
        return (f"{self.__class__.__name__}(agent_id={self.agent_id}, "
                f"model_type={self.model_type.value})")
