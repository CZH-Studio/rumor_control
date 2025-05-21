#!/usr/bin/env python
from crewai.flow.persistence.base import FlowPersistence
from typing import Any, Optional, Dict
from datetime import datetime  # Python 内置库
import sqlite3  # Python 内置库
import logging
import sys
import random
import warnings
import asyncio
from typing import List, Optional, Any  # Added Any
import json

from oasis.social_agent import AgentGraph, SocialAgent
from oasis.social_platform.typing import ActionType

from crewai.flow.flow import Flow, listen, router, start, FlowPersistence
# Added SQLiteFlowPersistence for default
from crewai.flow.persistence import persist, SQLiteFlowPersistence
from crewai import LLM

from rumor_control.src.rumor_control_group.crews.rumor_identify_crew.RumorIdentifyCrew import RumorIdentifyCrew
from rumor_control.src.rumor_control_group.crews.susceptibility_test_crew.SusceptibilityTestCrew import SusceptibilityTestCrew
from rumor_control.src.rumor_control_group.crews.recommend_predict_crew.RecommendPredictCrew import RecommendPredictCrew
from rumor_control.src.rumor_control_group.crews.rumor_refute_crew.rumor_refute_crew import rumor_refute_crew
from rumor_control.src.rumor_control_group.crews.rumor_inoculation_crew.rumor_inoculation_crew import rumor_inoculation_crew
from rumor_control.src.rumor_control_group.crews.broadcast_crew.broadcast_crew import broadcast_crew

from rumor_control.src.rumor_control_group.data_type.data_types import *
from rumor_control.src.rumor_control_group.data_type.constants import *


from crewai.flow.flow import Flow, listen, or_, router, start
from pydantic import BaseModel


warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

if "sphinx" not in sys.modules:
    agent_log = logging.getLogger(name="crewai.agent")
    agent_log.setLevel("DEBUG")
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_handler = logging.FileHandler(
        f"./crewai_log/crewai.agent-{str(now)}.log")
    file_handler.setLevel("DEBUG")
    file_handler.setFormatter(
        logging.Formatter(
            "%(levelname)s - %(asctime)s - %(name)s - %(message)s"))
    agent_log.addHandler(file_handler)


# custom_persistence.py (或者在 my_flow.py 顶部)

# 你需要确保这个导入路径对于你的 crewai 版本是正确的
# 通常它可能是 from crewai.core.flow.persistence.base import FlowPersistence
# 或者根据你的 crewai 安装结构调整
# 如果找不到 FlowPersistence，检查你的 crewai 版本和文档，
# 或者尝试继承 SQLiteFlowPersistence 并只修改序列化部分（但这更复杂）


class CustomSQLitePersistence(FlowPersistence):
    def __init__(self, db_path: str = "custom_flow_states.sqlite"):
        self.db_path = db_path
        self.init_db()  # 调用 init_db 来创建表

    def _get_connection(self):  # 这是一个辅助方法，保持不变
        return sqlite3.connect(self.db_path)

    # 重命名 _create_table_if_not_exists 为 init_db
    # 这个方法现在实现了 FlowPersistence 基类要求的抽象方法
    def init_db(self):
        conn = self._get_connection()
        cursor = conn.cursor()
        # 创建表的 SQL 语句，确保 UNIQUE 约束是 (flow_uuid, method_name)
        # 以便每个方法的状态都能被独立保存和查询
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS flow_states (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                flow_uuid TEXT NOT NULL,
                method_name TEXT NOT NULL,
                state_data TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(flow_uuid, method_name)
            )
            """
        )
        conn.commit()
        conn.close()

    def save_state(self, flow_uuid: str, method_name: str, state_data: Any):
        conn = self._get_connection()
        cursor = conn.cursor()
        final_state_dict_for_json = {}

        if isinstance(state_data, BaseModel):
            if hasattr(state_data, 'model_dump'):  # Pydantic V2
                final_state_dict_for_json = state_data.model_dump(mode='json')
            elif hasattr(state_data, 'dict'):  # Pydantic V1
                final_state_dict_for_json = state_data.dict()
            else:
                conn.close()
                raise TypeError(
                    f"Cannot serialize Pydantic model of type {type(state_data)} "
                    "lacking model_dump or dict method."
                )
        elif isinstance(state_data, dict):
            final_state_dict_for_json = state_data
        else:
            conn.close()
            raise TypeError(
                f"State data for persistence must be a Pydantic BaseModel or a dict, "
                f"got {type(state_data)}"
            )

        try:
            serialized_state = json.dumps(final_state_dict_for_json)
        except TypeError as e:
            conn.close()
            # 调试信息
            print(
                f"DEBUG: Failed to serialize final_state_dict_for_json. Content: {final_state_dict_for_json}")
            raise TypeError(
                f"Error serializing the processed state_dict to JSON: {e}. "
                f"Original state_data type: {type(state_data)}"
            ) from e

        cursor.execute(
            "INSERT OR REPLACE INTO flow_states (flow_uuid, method_name, state_data, timestamp) VALUES (?, ?, ?, ?)",
            (flow_uuid, method_name, serialized_state, datetime.now()),
        )
        conn.commit()
        conn.close()
        print(
            f"CustomSQLitePersistence: State for flow '{flow_uuid}', method '{method_name}' saved.")

    def load_state(self, flow_uuid: str, method_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
        conn = self._get_connection()
        cursor = conn.cursor()
        # crewai 的 load_state 可能需要 method_name 来加载特定方法的状态
        if method_name:
            cursor.execute(
                "SELECT state_data FROM flow_states WHERE flow_uuid = ? AND method_name = ? ORDER BY timestamp DESC LIMIT 1",
                (flow_uuid, method_name)
            )
        else:
            # 如果不提供 method_name，可以考虑加载与 flow_uuid 相关的最新状态，
            # 但这可能不符合 crewai 的预期行为。
            # crewai 的 @persist 装饰器在保存时会记录方法名，加载时也可能需要。
            # 为了安全起见，如果 method_name 为 None，可以返回 None 或抛出错误，
            # 或者根据 crewai 的具体期望调整此逻辑。
            # 这里我们假设如果 method_name 为 None，我们就不加载任何特定方法的状态。
            cursor.execute(
                "SELECT state_data FROM flow_states WHERE flow_uuid = ? ORDER BY timestamp DESC LIMIT 1",
                (flow_uuid,)  # 加载与 flow_uuid 关联的最新（不特定于方法）的状态，可能需要调整
            )
        row = cursor.fetchone()
        conn.close()
        if row:
            return json.loads(row[0])
        # print(f"CustomSQLitePersistence: load_state for flow '{flow_uuid}' (method: {method_name}) - No state found or method_name required.")
        return None








# Use method-level persistence
class RumorControlFlow(Flow[RumorInfectionState]):
    initial_state = RumorInfectionState

    # __init__ remains synchronous
    def __init__(
        self,
        private_territory: List[int],
        anchor_users: List[int],
        specialized_refute: bool,
        # agent_graph: AgentGraph,
        persistence: Optional[FlowPersistence] = None,  # Optional persistence
        rumor: str="",
        **kwargs: Any  # Accept other kwargs for Flow base class
    ):
        # Call super().__init__ first is good practice
        super().__init__(persistence=persistence, **kwargs)

        # Initialize state attributes after super init
        # self.state comes from Flow base or initial_state instantiation
        if self._state is None:  # Ensure state exists
            self._state = self.initial_state() if isinstance(
                self.initial_state, type) else self.initial_state

        self.state.anchor_users = anchor_users
        self.state.territory = set(anchor_users)  # Initialize territory
        self.specialized_refute = specialized_refute
        self.state.rumor_sources.append(RumorSource(post_id=1, user_id=55, content=rumor, refute="", topic="", created_timestep=0))

        # Initialize inspect_division properly
        if not hasattr(self.state, 'inspect_division') or self.state.inspect_division is None:
            self.state.inspect_division = {}

        for i in range(len(anchor_users)):
            anchor_id = anchor_users[i]
            if anchor_id not in self.state.inspect_division:
                # Create state if needed
                self.state.inspect_division[anchor_id] = InspectState()
            # Ensure private_territory is a set
            if not isinstance(self.state.inspect_division[anchor_id].private_territory, set):
                self.state.inspect_division[anchor_id].private_territory = set(
                )
            self.state.inspect_division[anchor_id].private_territory.add(
                private_territory[i])
            self.state.inspect_division[anchor_id].userstate.susceptible.add(
                private_territory[i])

        self.llm = LLM(model="glm-4-flash")
        self.state.rumor_sources.append(RumorSource(
            post_id=1, user_id=1, content="report: amazon plans to open its first physical store, in new york URL", refute="", topic="business", created_timestep=0))
        agent_log.info(
            f"RumorControlFlow initialized with anchor_users: {anchor_users}, private_territory: {private_territory}")
        # self._current_agent_graph: Optional[AgentGraph] = agent_graph

    def set_agent_graph(self, agent_graph: AgentGraph, current_timestep: int):
        """在流程执行前，替换当前的 agent_graph"""
        self._current_agent_graph = agent_graph
        self._current_timestep = current_timestep

    
    
    
    
    
    
    
    
    @start()
    @persist(CustomSQLitePersistence(db_path="rumor_flow_data.sqlite"))
    async def detect(self):
        # agent_graph = self.state.agent_graph
        if self._current_agent_graph is None:
            print("Error: agent_graph not found in flow state during detect.")
            return

        agent_log.info(f"timestep {self._current_timestep}")

        activated = set()
        for agent in self.state.inspect_division:
            if self.state.inspect_division[agent].triggered:
                activated.add(agent)
        agent_log.info(f"activated anchor users: {activated}")
        agent_log.info(f"start detecting rumors...")

        self.state.refute_applyment = set()
        self.state.inoculation_applyment = set()
        newly_infected = dict()

        anchor_users = self._current_agent_graph.get_agents_by_ids(
            self.state.anchor_users)
        if not anchor_users:
            print("Warning: No anchor users found in the graph.")
            return

        union_set = set()
        self.state.post_refute = set()

        for anchor_user in anchor_users:
            anchor_id = anchor_user.agent_id
            print(f"anchor_user_{anchor_id} receive message:")
            # Assuming env.to_text_prompt is synchronous or fast
            context_data = await anchor_user.env.get_posts_list()
            current_user_state = self.state.inspect_division[anchor_id].userstate
            poster_ids = [int(post.get("user_id")) for post in context_data]
            # agent_log.info(f"anchor_user_{anchor_id} received context data with poster_ids: {poster_ids}")

            # Ensure sets exist before modifying
            current_user_state.susceptible = current_user_state.susceptible
            current_user_state.infected = current_user_state.infected
            current_user_state.recovered = current_user_state.recovered
            current_user_state.still_infected = current_user_state.still_infected
            # --- is_rumor (can remain sync helper or integrate) ---
            async def is_rumor_async(post_content):
                if "Original Content: " in post_content:
                    processed_content = post_content.split(
                        "Original Content: ", 1)[-1]
                elif "Quote content: " in post_content:
                    processed_content = post_content.split(
                        "Quote content: ", 1)[-1]
                elif "Repost content: " in post_content:
                    processed_content = post_content.split(
                        "Repost content: ", 1)[-1]
                else:
                    processed_content = post_content
                if processed_content in [r.content for r in self.state.rumor_sources]:
                    return "old"
                try:
                    # Ensure input is just the content string
                    crew_input = {"post_content": processed_content, "rumor_sources": [
                        source.content for source in self.state.rumor_sources]}
                    print(
                        f"Running RumorIdentifyCrew with input: {crew_input}")
                    result = await RumorIdentifyCrew().crew().kickoff_async(inputs=crew_input)
                    print(f"RumorIdentifyCrew result: {result}")
                    if result.raw.strip().lower() == "yes":
                        return "new"
                    else:
                        return "not_rumor"
                except Exception as e:
                    print(f"Error during RumorIdentifyCrew kickoff: {e}")
                    return "not_rumor"  # Default to not_rumor on error

            

            # Call the sync helper
            identify_task = []
            for post in context_data:
                if not isinstance(post, dict):  # Ensure post is a dict
                    print(
                        f"Skipping invalid post item for anchor_user_{anchor_id}: {post}")
                    continue
                if int(post.get("user_id")) == anchor_id:  # Skip own post
                    continue
                identify_task.append(is_rumor_async(post.get("content")))
            rumor_status_list = await asyncio.gather(*identify_task, return_exceptions=True)
            for rumor_status, post in (zip(rumor_status_list, context_data)):
                if isinstance(rumor_status, Exception):
                    print(f"Error during is_rumor_async: {rumor_status}")
                    rumor_status = "not_rumor"  # Default to not_rumor on error

                print("post: ", json.dumps(post, indent=2))
                user_id = int(post.get("user_id"))
                content = post.get("content", "")
                post_id = post.get("post_id")

                print(f"Rumor status for post {post_id}: {rumor_status}")
                agent_log.info(
                    f"anchor_user_{anchor_id} received post {post_id} from user_{user_id} with rumor_status: {rumor_status}")
                
                if rumor_status != "not_rumor":  # 已有谣言
                    self.state.post_refute.add((post_id, user_id, content, "easy to share misinformation"))
                
                if rumor_status != "not_rumor":
                    if user_id not in union_set and user_id not in self.state.territory:
                        if not self.state.inspect_division[anchor_id].triggered:
                            self.state.inspect_division[anchor_id].triggered = True
                            agent_log.info(f"anchor_user_{anchor_id} triggered! -")
                            print(f"anchor_user_{anchor_id} triggered!")
                        current_user_state.susceptible.add(user_id)
                        union_set.add(user_id)  # 加入选择列表

                # 易感者发帖
                if user_id in current_user_state.susceptible:
                    if rumor_status != "not_rumor":
                        if rumor_status == "new":
                            # Using await for potential async LLM call
                            llm_call_needed = True  # Set flag if LLM needed
                            if llm_call_needed and hasattr(self, 'llm') and self.llm:
                                try:
                                    response = await self.llm.acall(
                                        f"you see a post: {json.dumps(post)}. Categorize the topic of the post. don't output anything else."
                                    )
                                    topic = response.choices[0].message.content
                                    print("rumor topic: ", topic)
                                except Exception as e:
                                    print(f"LLM call failed: {e}")
                                    topic = "news"
                            else:
                                # Fallback or skip topic generation if LLM not available/needed
                                print("Skipping LLM topic generation.")
                                topic = "news"  # Assign default topic
                            self.state.rumor_sources.append(RumorSource(
                                post_id=post_id, user_id=user_id, content=content, refute="", topic=topic, created_timestep=self._current_timestep))

                        current_user_state.susceptible.discard(user_id)
                        current_user_state.infected.add(user_id)

                        # 其所有关注者也感染
                        g = self._current_agent_graph.graph
                        try:
                            # Ensure user_id exists as a vertex
                            # Assuming node names are strings
                            user_vertex = g.vs.find(name=str(user_id))
                            # Find followers (edges pointing TO user_vertex)
                            incoming_edges = g.es.select(
                                _target=user_vertex.index)
                            followers = [int(g.vs[edge.source]["name"]) for edge in incoming_edges] or [
                            ]  # Convert name back to int
                            # agent_log.info(
                            #     f"infected_user_{user_id} found followers: {followers}")

                            newly_infected_followers = set(
                                followers) - current_user_state.infected - current_user_state.recovered - current_user_state.still_infected
                            current_user_state.infected.update(
                                newly_infected_followers)

                            # Newly infected = followers not already in territory + the user themselves
                            new_infections_for_territory = (
                                newly_infected_followers - self.state.territory)
                            newly_infected[anchor_id] = (
                                list(new_infections_for_territory), user_id)
                            # print(f"Newly infected: {self.state.newly_infected}")

                            # Add user to refute list
                            self.state.refute_applyment.add(user_id)
                        except ValueError:
                            print(
                                f"User ID {user_id} not found in graph vertices.")
                        except Exception as e:
                            print(
                                f"Error processing followers for user {user_id}: {e}")

                    else:  # 如果易感者发帖不再涉及谣言，则视为康复
                        current_user_state.susceptible.discard(user_id)
                        current_user_state.recovered.add(user_id)

                # 感染者发帖
                elif user_id in current_user_state.infected:
                    if rumor_status == "not_rumor":
                        current_user_state.infected.discard(user_id)
                        current_user_state.recovered.add(user_id)
                    else:  # 如果继续转发谣言
                        current_user_state.infected.discard(user_id)
                        current_user_state.still_infected.add(
                            user_id)  # 辟谣次数用光，不再进行辟谣

                # 未能治愈者发帖
                elif user_id in current_user_state.still_infected:
                    if rumor_status == "not_rumor":
                        current_user_state.still_infected.discard(user_id)
                        current_user_state.recovered.add(user_id)
                    # else: Still infected, no change

                # 康复者发帖
                elif user_id in current_user_state.recovered:
                    # TODO: Could re-infect? Or promote refutation?
                    pass  # No action for recovered users posting for now

            agent_log.info(f"anchor_user_{anchor_id}: {current_user_state}")

        # TODO: 局势分析crew,用于决定broadcastCrew名单
        return newly_infected

    # @router(detect)
    # @persist(CustomSQLitePersistence(db_path="rumor_flow_data.sqlite"))
    # async def if_triggered(self):
    #     for anc in self.state.anchor_users:
    #          # Check if anchor exists and is triggered
    #          if anc in self.state.inspect_division and self.state.inspect_division[anc].triggered:
    #             print("detect finished, ready to select. ")
    #             return "activated"
    #     print("no rumor detected. ")
    #     return "silent"

    
    
    
    
    
    
    
    
    @listen(detect)
    @persist(CustomSQLitePersistence(db_path="rumor_flow_data.sqlite"))
    async def select(self, newly_infected):
        if self._current_agent_graph is None:
            print(
                "Error: agent_graph not available in select state (was not passed to detect).")
            return None, None

        agent_log.info(f"start selecting users...")

        ref_apply_final = set()
        inoc_apply_final = set()

        processed_anchors = set()  # Track anchors whose territory has been processed this round

        ref_apply_result_with_reason = {}
        inoc_apply_result_with_reason = {}

        for anchor_user_obj in self._current_agent_graph.get_agents_by_ids(self.state.anchor_users):
            anchor_id = anchor_user_obj.agent_id

            # Skip if anchor not triggered or already processed its part
            if not self.state.inspect_division[anchor_id].triggered or anchor_id in processed_anchors:
                continue
            if not newly_infected.get(anchor_id):
                continue  # Skip if no newly infected users for anchor

            newly_infected_tuple = newly_infected[anchor_id]
            newly_infected_followers = []
            if isinstance(newly_infected_tuple[0], (list, set)):
                newly_infected_followers = list(newly_infected_tuple[0])
            # The user who posted the rumor (might be None if only followers processed)
            infected_poster_id = newly_infected_tuple[1] if len(
                newly_infected_tuple) > 1 else None

            ref_apply_result_with_reason[anchor_id] = {}
            inoc_apply_result_with_reason[anchor_id] = {}

            print(f"Processing selection for anchor {anchor_id}")
            processed_anchors.add(anchor_id)  # Mark as processed

            # Combine followers and potentially the original poster for susceptibility test
            users_to_test_ids = set(newly_infected_followers) - \
                self.state.inspect_division[anchor_id].private_territory

            # --- Refute Selection ---
            print("selecting refute users ...")
            profiles_for_refute = []
            for user_id in users_to_test_ids:
                agent = self._current_agent_graph.get_agent(user_id)
                if agent and hasattr(agent, 'user_info') and hasattr(agent.user_info, 'description'):
                    profiles_for_refute.append({
                        "user_id": user_id,
                        "user_profile": agent.user_info.description,
                    })
                else:
                    print(f"Warning: Could not get profile for user {user_id}")

            if profiles_for_refute:
                # Refute at least 1 if possible
                num_to_refute = max(1, int(0.2 * len(profiles_for_refute)))
                try:
                    # NOTE: Crew kickoff is sync. Consider async if available.
                    ref_apply_crew_input = {
                        "num": num_to_refute,
                        "candidates": list(users_to_test_ids),
                        "personal_profile": profiles_for_refute,
                    }
                    print(
                        f"Running SusceptibilityTestCrew (refute) with input: {ref_apply_crew_input}")
                    ref_apply_result = SusceptibilityTestCrew().crew().kickoff(
                        inputs=ref_apply_crew_input)  # Pass as dict
                    ref_apply_result = json.loads(ref_apply_result.raw)
                    print(
                        f"SusceptibilityTestCrew (refute) result: {ref_apply_result}")
                    # agent_log.info(
                        # f"anchor_user_{anchor_id}: refutation selection result: {ref_apply_result}")

                    if isinstance(ref_apply_result, list):
                        ref = set()
                        for ref_with_reason in ref_apply_result:
                            if ref_with_reason["user_id"] in users_to_test_ids:
                                ref.add(ref_with_reason["user_id"])
                                ref_apply_result_with_reason[anchor_id][ref_with_reason["user_id"]
                                                                        ] = ref_with_reason["reason"]
                        agent_log.info(
                            f"anchor_user_{anchor_id}: refutation selected: {ref}")
                        ref.add(infected_poster_id)
                        ref_apply_final.update(ref)
                        # Update territory only with users selected for refutation
                        self.state.territory.update(ref)
                        # Also update the specific anchor's private territory
                        self.state.inspect_division[anchor_id].private_territory.update(
                            ref)
                    else:
                        print(
                            "Warning: SusceptibilityTestCrew (refute) did not return a list.")
                        agent_log.info(
                            "Warning: SusceptibilityTestCrew (refute) did not return a list.")

                except Exception as e:
                    print(
                        f"Error during SusceptibilityTestCrew (refute) kickoff: {e}")
                    agent_log.info(
                        f"anchor_user_{anchor_id}: refutation selection failed with error: {e}")
            else:
                print("No profiles available for refutation selection.")
                agent_log.info(
                    f"anchor_user_{anchor_id}: no refutation selected.")

            # --- Inoculation Selection ---
            print("selecting inoculation users ...")
            g = self._current_agent_graph.graph
            susceptible_followers_to_inoculate = set()

            # Find followers of the newly infected users (original list + poster)
            for user_id in users_to_test_ids:
                try:
                    user_vertex = g.vs.find(name=str(user_id))
                    # Find followers (edges pointing TO user_vertex)
                    incoming_edges = g.es.select(_target=user_vertex.index)
                    followers_of_infected = {
                        int(g.vs[edge.source]["name"]) for edge in incoming_edges}
                    # Add followers who are not already in territory
                    susceptible_followers_to_inoculate.update(
                        followers_of_infected - self.state.territory)
                except ValueError:
                    print(
                        f"User ID {user_id} not found in graph for inoculation follower search.")
                except Exception as e:
                    print(
                        f"Error finding followers of {user_id} for inoculation: {e}")
                    agent_log.info(
                        f"anchor_user_{anchor_id}: error finding followers of {user_id} for inoculation: {e}")

            profiles_for_inoculation = []
            for user_id in susceptible_followers_to_inoculate:
                agent = self._current_agent_graph.get_agent(user_id)
                if agent and hasattr(agent, 'user_info') and hasattr(agent.user_info, 'description'):
                    profiles_for_inoculation.append({
                        "user_id": user_id,
                        "user_profile": agent.user_info.description,
                    })
                else:
                    print(
                        f"Warning: Could not get profile for user {user_id} (inoculation)")

            if profiles_for_inoculation:
                # Inoculate at least 1
                num_to_inoculate = max(
                    1, int(0.2 * len(profiles_for_inoculation)))
                try:
                    inoc_apply_crew_input = {
                        "num": num_to_inoculate,
                        "candidates": list(susceptible_followers_to_inoculate),
                        "personal_profile": profiles_for_inoculation,
                    }
                    print(
                        f"Running SusceptibilityTestCrew (inoculation) with input: {inoc_apply_crew_input}")
                    inoc_apply_result = SusceptibilityTestCrew().crew().kickoff(
                        inputs=inoc_apply_crew_input)  # Pass as dict
                    inoc_apply_result = json.loads(inoc_apply_result.raw)
                    # agent_log.info(
                    #     f"anchor_user_{anchor_id}: inoculation selection result: {inoc_apply_result}")
                    print(
                        f"SusceptibilityTestCrew (inoculation) result: {inoc_apply_result}")

                    if isinstance(inoc_apply_result, list):
                        inoc = set()
                        for inoc_with_reason in inoc_apply_result:
                            # 合法性检查
                            if inoc_with_reason["user_id"] in susceptible_followers_to_inoculate:
                                inoc.add(inoc_with_reason["user_id"])
                                inoc_apply_result_with_reason[anchor_id][inoc_with_reason["user_id"]
                                                                         ] = inoc_with_reason["reason"]
                        inoc_apply_final.update(inoc)
                        # Update territory
                        self.state.territory.update(inoc)
                        self.state.inspect_division[anchor_id].private_territory.update(
                            inoc)
                        agent_log.info(
                            f"anchor_user_{anchor_id}: inoculation selected: {inoc}")
                    else:
                        print(
                            "Warning: SusceptibilityTestCrew (inoculation) did not return a list.")
                        agent_log.info(
                            "Warning: SusceptibilityTestCrew (inoculation) did not return a list.")
                except Exception as e:
                    print(
                        f"Error during SusceptibilityTestCrew (inoculation) kickoff: {e}")
                    agent_log.info(
                        f"anchor_user_{anchor_id}: inoculation selection failed with error: {e}")
            else:
                print("No profiles available for inoculation selection.")
                agent_log.info(
                    f"anchor_user_{anchor_id}: no inoculation selected.")

            # --- Cross-Domain Inoculation (Simplified/Corrected Logic) ---
            print("selecting cross-domain inoculation users ...")
            agent_log.info(
                f"anchor_user_{anchor_id}: cross-domain inoculation selection...")
            candidates_for_cross_domain = set()
            candidate_to_anchor_map = {}

            current_anchor_agent = self._current_agent_graph.get_agent(
                anchor_id)

            # Find the rumor source related to the current infection event if possible
            # This logic assumes the first rumor source is relevant, might need refinement
            current_rumor = self.state.rumor_sources[0]

            other_anchor_ids = set(self.state.anchor_users) - {anchor_id}

            for other_anchor_id in other_anchor_ids:
                if other_anchor_id in self.state.inspect_division:
                    other_anchor_state = self.state.inspect_division[other_anchor_id]
                    # Check if the other anchor is NOT triggered (or has very few susceptible - adjust criteria)
                    if not other_anchor_state.triggered or len(other_anchor_state.private_territory) < 3:
                        # Select some users from the other anchor's territory as candidates
                        # Using private_territory, limit the number
                        #   potential_cands = list(other_anchor_state.private_territory - self.state.territory)
                        #   num_cands_to_take = min(5, len(potential_cands)) # Take up to 5 candidates
                        #   selected_cands = random.sample(potential_cands, num_cands_to_take) if potential_cands else []

                        #   for cand_id in selected_cands:
                        #       candidates_for_cross_domain.append(cand_id)
                        #       candidate_to_anchor_map[cand_id] = other_anchor_id # Map candidate back to its original anchor
                        cand_ids = other_anchor_state.private_territory
                        candidates_for_cross_domain.update(cand_ids)
                        for cand_id in cand_ids:
                            candidate_to_anchor_map[cand_id] = other_anchor_id
                else:
                    print(f"No state found for other anchor {other_anchor_id}")

            agent_log.info(
                f"anchor_user_{anchor_id}: cross-domain inoculation candidates: {candidates_for_cross_domain}")
            if candidates_for_cross_domain:
                receivers_profiles = []
                for rec_id in candidates_for_cross_domain:
                    rec_agent = self._current_agent_graph.get_agent(rec_id)
                    receivers_profiles.append(
                        {"user_id": rec_id, "user_profile": rec_agent.user_info.description})
                if receivers_profiles:
                    # Predict for at least 1
                    num_cross_domain = int(0.5 * len(receivers_profiles))
                    poster_agent = self._current_agent_graph.get_agent(
                        current_rumor.user_id)  # Agent who posted the rumor
                    poster_profile = poster_agent.user_info.description if poster_agent else "Unknown"

                    try:
                        rec_predict_input = {
                            "num": num_cross_domain,
                            "rec_type": current_anchor_agent.user_info.recsys_type if hasattr(current_anchor_agent.user_info, 'recsys_type') else "default",
                            "candidates": list(candidates_for_cross_domain),
                            "user_profile": poster_profile,
                            "post": current_rumor.content,
                            "fan_count": self._current_agent_graph.graph.degree(current_rumor.user_id, mode="in"),
                            "current_timestamp": self._current_timestep,
                            "created_timestamp": current_rumor.created_timestep,
                            "recievers": receivers_profiles,
                        }
                        print(
                            f"Running RecommendPredictCrew with input: {json.dumps(rec_predict_input, indent=2)}")
                        rec_apply_far_result = RecommendPredictCrew().crew().kickoff(
                            inputs=rec_predict_input)  # Pass as dict
                        rec_apply_far_result = json.loads(
                            rec_apply_far_result.raw)
                        print(
                            f"RecommendPredictCrew result: {rec_apply_far_result}")
                        agent_log.info(
                            f"anchor_user_{anchor_id}: cross-domain inoculation selected: {rec_apply_far_result}")

                        # Expecting list of user IDs directly based on previous logic? Adapt if needed.
                        if isinstance(rec_apply_far_result, list):
                            # Assuming result is list of user IDs
                            rec_apply_far_ids = set(rec_apply_far_result)
                            # Add to final inoculation set
                            inoc_apply_final.update(rec_apply_far_ids)
                            self.state.territory.update(rec_apply_far_ids)

                            # Update the territories of the *original* anchors of these candidates
                            for rec_cand_id in rec_apply_far_ids:
                                original_anchor_id = candidate_to_anchor_map.get(
                                    rec_cand_id)
                                # 合法性检查
                                if original_anchor_id and original_anchor_id in self.state.inspect_division:
                                    self.state.inspect_division[original_anchor_id].private_territory.add(
                                        rec_cand_id)
                                    # 中途加入需要插入逻辑
                                    # inoc_apply_result_with_reason[original_anchor_id] = {
                                    # }
                                    # inoc_apply_result_with_reason[original_anchor_id][rec_cand_id] = ""
                                    # Activate the other anchor if it wasn't already
                                    if not self.state.inspect_division[original_anchor_id].triggered:
                                        print(
                                            f"Activating anchor {original_anchor_id} due to cross-domain inoculation.")
                                        self.state.inspect_division[original_anchor_id].triggered = True
                                        agent_log.info(
                                            f"anchor_user_{original_anchor_id} triggered! -")

                        else:
                            print(
                                "Warning: RecommendPredictCrew did not return a list.")
                            agent_log.info(
                                "Warning: RecommendPredictCrew did not return a list.")

                    except Exception as e:
                        print(
                            f"Error during RecommendPredictCrew kickoff: {e}")
                        agent_log.info(
                            f"anchor_user_{anchor_id}: Error during RecommendPredictCrew kickoff: {e}")
                else:
                    print("No receiver profiles for cross-domain prediction.")
                    agent_log.info(
                        "No receiver profiles for cross-domain prediction.")
            else:
                print("No candidates found for cross-domain inoculation.")
                agent_log.info(
                    f"anchor_user_{anchor_id}: no cross-domain inoculation selected.")

        # Update final sets after processing all anchors
        self.state.refute_applyment = ref_apply_final
        self.state.inoculation_applyment = inoc_apply_final

        print(
            f"Select finished. Refute Set: {self.state.refute_applyment}, Inoculation Set: {self.state.inoculation_applyment}")
        # Return structure might need adjustment based on how @listen uses it,
        # but returning the final sets seems logical.
        return ref_apply_result_with_reason, inoc_apply_result_with_reason

    
    
    
    
    
    
    
    
    @listen(select)
    @persist(CustomSQLitePersistence(db_path="rumor_flow_data.sqlite"))
    async def refute(self, select_output):

        ref_apply_result_with_reason, inoc_apply_result_with_reason = select_output

        if self._current_agent_graph is None:
            print(
                "Error: agent_graph not available in refute state (was not passed to detect).")
            return

        print("Refute/Inoculate phase started...")
        agent_log.info("Refute/Inoculate phase started...")
        # Get relevant rumor (assuming first one, may need better logic)
        rumor_to_address = self.state.rumor_sources[0] if self.state.rumor_sources else None
        if not rumor_to_address:
            print("No rumor source identified for refutation/inoculation.")
            return

        # Generate refute text once if needed
        refute_text = rumor_to_address.refute
        if not refute_text:
            print("Generating generic refute text...")
            try:
                # NOTE: Crew kickoff sync.
                # Need a representative user or context for generic refute
                refute_crew_input = {
                    # Provide necessary context based on crew needs
                    "profile": REPRESENTATIVE_USER_PROFILE,  # If needed
                    "reason_been_chosen": REPRESENTATIVE_REASON_BEEN_CHOSEN,
                    "rumor": rumor_to_address.content,
                    "topic": rumor_to_address.topic,
                }
                print(
                    f"Running rumor_refute_crew with input: {refute_crew_input}")
                generated_refute = rumor_refute_crew().crew().kickoff(
                    inputs=refute_crew_input)  # Pass as dict
                generated_refute = generated_refute.raw
                print(f"rumor_refute_crew result: {generated_refute}")
                if isinstance(generated_refute, str):
                    refute_text = generated_refute
                    # Optionally store back to rumor_source if desired
                    rumor_to_address.refute = refute_text
                else:
                    print(
                        "Warning: Refute crew did not return a string. Using default.")
                    refute_text = DEFAULT_REFUTATION
            except Exception as e:
                print(f"Error during refute crew kickoff, using default: {e}")
                # Default on error
                refute_text = DEFAULT_REFUTATION

        # processed_users_for_action = set() # Track users already acted upon this cycle

        # Iterate through anchors to perform actions within their territories
        follow_tasks = []
        refute_tasks = []
        inoculate_tasks = []
        post_refute_tasks = []
        dislike_tasks = []

        all_refute_users = []
        all_inoculate_users = []
        for anchor_user_obj in self._current_agent_graph.get_agents_by_ids(self.state.anchor_users):
            anchor_id = anchor_user_obj.agent_id
            if anchor_id not in self.state.inspect_division or not self.state.inspect_division[anchor_id].triggered:
                continue  # Skip inactive anchors

            anchor_territory = self.state.inspect_division[anchor_id].private_territory

            # --- Follow Action (Optional, if needed before messaging) ---
            users_to_potentially_follow = (
                self.state.refute_applyment | self.state.inoculation_applyment) & anchor_territory
            
            for followee_id in users_to_potentially_follow:
                # Check if already following? Assume env handles this.
                args = {"followee_id": followee_id}
                print(f"Anchor {anchor_id} attempting to follow {followee_id}")
                # Assuming follow action is async
                follow_tasks.append(
                    getattr(anchor_user_obj.env.action, ActionType.FOLLOW.value)(**args))
        

            # --- Refutation Actions ---
            users_to_refute_in_territory = (
                self.state.refute_applyment & anchor_territory)
            all_refute_users += list(users_to_refute_in_territory)
            agent_log.info(
                f"anchor_user_{anchor_id}: refuting users: {users_to_refute_in_territory}")
            gen_refute_text = ""
            print(
                f"Anchor {anchor_id} refuting users: {users_to_refute_in_territory}")
            for infected_user_id in users_to_refute_in_territory:
                infected_user_agent = self._current_agent_graph.get_agent(
                    infected_user_id)
                if self.specialized_refute:
                    try:
                        # NOTE: Crew kickoff sync. Generate personalized refute.
                        ref_crew_input = {
                            "profile": infected_user_agent.user_info.description,
                            "reason_been_chosen": ref_apply_result_with_reason[anchor_id][infected_user_id] if anchor_id in ref_apply_result_with_reason.keys() else "easy to share misinformation",
                            "rumor": rumor_to_address.content,
                            "topic": rumor_to_address.topic,
                        }
                        print(
                            f"Running rumor_refute_crew for {infected_user_id}")
                        gen_refute_text = rumor_refute_crew().crew().kickoff(
                            inputs=ref_crew_input)  # Pass as dict
                        gen_refute_text = gen_refute_text.raw
                        print(f"rumor_refute_crew result: {gen_refute_text}")
                        # agent_log.info(f"anchor_user_{anchor_id}: generated refute for user_{infected_user_id}: {gen_refute_text}")

                        if not isinstance(gen_refute_text, str):
                            print(
                                "Warning: refutation crew did not return string. Using default.")
                            gen_refute_text = DEFAULT_REFUTATION
                            agent_log.info(
                                "Warning: refutation crew did not return string. Using default.")
                        else:
                            agent_log.info(
                                f"anchor_user_{anchor_id}: creating refute post targeting user_{infected_user_id}: {gen_refute_text}")
                    except Exception as e:
                        print(
                            f"Error during refutation crew kickoff for {infected_user_id}: {e}")
                        agent_log.info(
                            f"anchor_user_{anchor_id}: refute generation failed with error, using default: {e}")
                        gen_refute_text = refute_text

                else:
                    # Send refute as a post or direct message? Assuming CREATE_POST for broad reach
                    gen_refute_text = refute_text
                args = {"content": gen_refute_text}
                print(
                    f"Anchor {anchor_id} creating refute post targeting user context (indirectly) {infected_user_id}")
                # Action likely async
                refute_tasks.append(
                    self.perform_env_action(
                        anchor_user_obj, ActionType.CREATE_POST, args, target_user_id=infected_user_id)
                )

                           
            #  processed_users_for_action.update(users_to_refute_in_territory) # Mark as processed
            # Process results if needed (e.g., store post_id)

            # --- Inoculation Actions ---
            users_to_inoculate_in_territory = (
                self.state.inoculation_applyment & anchor_territory)  # - processed_users_for_action
            all_inoculate_users += list(users_to_inoculate_in_territory)
            agent_log.info(
                f"anchor_user_{anchor_id}: inoculating users: {users_to_inoculate_in_territory}")
            gen_inoculation_text = ""
            print(
                f"Anchor {anchor_id} inoculating users: {users_to_inoculate_in_territory}")
            for sus_user_id in users_to_inoculate_in_territory:
                sus_user_agent = self._current_agent_graph.get_agent(
                    sus_user_id)
                try:
                    # NOTE: Crew kickoff sync. Generate personalized inoculation.
                    inoc_crew_input = {
                        "profile": sus_user_agent.user_info.description,
                        "reason_been_chosen": inoc_apply_result_with_reason[anchor_id][sus_user_id] if anchor_id in inoc_apply_result_with_reason.keys() else "",
                        "rumor": rumor_to_address.content,
                        "topic": rumor_to_address.topic,
                    }
                    print(f"Running rumor_inoculation_crew for {sus_user_id}")
                    gen_inoculation_text = rumor_inoculation_crew().crew().kickoff(inputs=inoc_crew_input) # Pass as dict
                    gen_inoculation_text = gen_inoculation_text.raw
                    print(f"rumor_inoculation_crew result: {gen_inoculation_text}")

                    if not isinstance(gen_inoculation_text, str):
                        print("Warning: Inoculation crew did not return string. Using default.")
                        agent_log.info("Warning: Inoculation crew did not return string. Using default.")
                        gen_inoculation_text = DEFAULT_INOCULATION
                    else:
                        agent_log.info(f"anchor_user_{anchor_id}: creating inoculation post targeting user_{sus_user_id}: {gen_inoculation_text}")
                except Exception as e:
                    print(f"Error during inoculation crew kickoff for {sus_user_id}: {e}")
                    gen_inoculation_text = DEFAULT_INOCULATION
                    agent_log.info(f"anchor_user_{anchor_id}: inoculation generation failed with error, using default: {e}")
                # Post inoculation message
                args = {"content": gen_inoculation_text}
                agent_log.info(
                    f"anchor_user_{anchor_id}: creating inoculation post targeting user_{sus_user_id}: {gen_inoculation_text}")
                print(
                    f"Anchor {anchor_id} creating inoculation post targeting user context (indirectly) {sus_user_id}")
                # Action likely async
                inoculate_tasks.append(
                    self.perform_env_action(
                        anchor_user_obj, ActionType.CREATE_POST, args, target_user_id=sus_user_id)
                )

            #  processed_users_for_action.update(users_to_inoculate_in_territory) # Mark as processed
            # Process results if needed

            # --- Broadcast Action (If anchor is designated) ---
            if anchor_id in self.state.broadcast_anchors.keys():
                print(f"Anchor {anchor_id} performing broadcast...")
                try:
                    # NOTE: Crew kickoff sync.
                    broadcast_crew_input = {
                        "user": {"id": anchor_id, "profile": anchor_user_obj.user_info.description if hasattr(anchor_user_obj, 'user_info') else ""},
                        "rumor": rumor_to_address.content,
                    }
                    print("Running broadcast_crew...")
                    gen_broadcast_text = broadcast_crew().crew().kickoff(
                        inputs=broadcast_crew_input)  # Pass as dict
                    gen_broadcast_text = gen_broadcast_text.raw
                    print(f"broadcast_crew result: {gen_broadcast_text}")

                    if not isinstance(gen_broadcast_text, str):
                        print(
                            "Warning: Broadcast crew did not return string. Skipping broadcast.")
                        continue

                except Exception as e:
                    print(f"Error during broadcast crew kickoff: {e}")
                    continue  # Skip broadcast on error

                args = {"content": gen_broadcast_text}
                broadcast_post_result = await self.perform_env_action(anchor_user_obj, ActionType.CREATE_POST, args)

                # If post created successfully, trigger likes from other anchors
                if broadcast_post_result and isinstance(broadcast_post_result, dict) and "post_id" in broadcast_post_result:
                    post_id = broadcast_post_result["post_id"]
                    print(
                        f"Broadcast post created: {post_id}. Triggering likes...")
                    like_tasks = []
                    other_anchor_ids = set(
                        self.state.anchor_users) - {anchor_id}
                    for other_anchor_id in other_anchor_ids:
                        other_anchor_agent = self._current_agent_graph.get_agent(
                            other_anchor_id)
                        if other_anchor_agent:
                            like_args = {"post_id": post_id}
                            like_tasks.append(
                                self.perform_env_action(
                                    other_anchor_agent, ActionType.LIKE_POST, like_args)
                            )
                    # Wait for likes
                    await asyncio.gather(*like_tasks, return_exceptions=True)

         # --- Refutation Actions (for posts) ---
        agent_log.info(
            f"refuting posts...")
        for post_refute in self.state.post_refute:
            rumor_post_agent = self._current_agent_graph.get_agent(
                post_refute[1])
            if self.specialized_refute:
                try:
                    # NOTE: Crew kickoff sync. Generate personalized refute.
                    ref_crew_input = {
                        "profile": rumor_post_agent.user_info.description,
                        "reason_been_chosen": post_refute[3],
                        "rumor": post_refute[2],
                        "topic": "news",
                    }
                    gen_refute_text = rumor_refute_crew().crew().kickoff(
                        inputs=ref_crew_input)  # Pass as dict
                    gen_refute_text = gen_refute_text.raw
                    print(f"rumor_refute_crew result: {gen_refute_text}")
                    # agent_log.info(f"anchor_user_{anchor_id}: generated refute for user_{infected_user_id}: {gen_refute_text}")

                    if not isinstance(gen_refute_text, str):
                        print(
                            "Warning: refutation crew did not return string. Using default.")
                        gen_refute_text = "your post is totally fake! Stay critical!"
                        agent_log.info(
                            "Warning: refutation crew did not return string. Using default.")
                    else:
                        agent_log.info(
                            f"anchor_user_{anchor_id}: creating refute targeting post {post_refute[0]}: {gen_refute_text}")
                except Exception as e:
                    print(
                        f"Error during refutation crew kickoff for post {post_refute[0]}: {e}")
                    agent_log.info(
                        f"anchor_user_{anchor_id}: refute generation failed with error, using default: {e}")
                    gen_refute_text = "your post is totally fake! Stay critical!"

            else:
                # Send refute as a post or direct message? Assuming CREATE_POST for broad reach
                gen_refute_text = "your post is totally fake! Stay critical!"
            args = {"post_id":post_refute[0],"content": gen_refute_text}
            # Action likely async
            post_refute_tasks.append(
                self.perform_env_action(
                    anchor_user_obj, ActionType.CREATE_COMMENT, args, target_user_id=post_refute[1])
            )
            team_mates = set(self.state.anchor_users)
            for team_mate in team_mates:
                dislike_tasks.append(
                    self.perform_env_action(
                        self._current_agent_graph.get_agent(team_mate), ActionType.DISLIKE_POST, {"post_id":post_refute[0]}, target_user_id=post_refute[1])
                )

        # 所有任务一起并发执行
        try:
            await asyncio.gather(*follow_tasks)
            # print(f"Anchor {anchor_id} finished follow attempts.")
        except Exception as e:
            pass
            # print(
            #     f"Error during follow actions for anchor {anchor_id}: {e}")
            # agent_log.info(
            #     f"anchor_user_{anchor_id}: follow action failed with error: {e}")
        
        results_refute = await asyncio.gather(*refute_tasks, return_exceptions=True)
        infected_user_agents = [self._current_agent_graph.get_agent(
            infected_user_id) for infected_user_id in all_refute_users]
        for i, pid in enumerate([re["post_id"] for re in results_refute if isinstance(re, dict) and "post_id" in re]):
            infected_user_agents[i].private_message_id = pid

        results_inoc = await asyncio.gather(*inoculate_tasks, return_exceptions=True)
        sus_user_agents = [self._current_agent_graph.get_agent(
            sus_user_id) for sus_user_id in all_inoculate_users]
        for i, pid in enumerate([re["post_id"] for re in results_inoc if isinstance(re, dict) and "post_id" in re]):
            sus_user_agents[i].private_message_id = pid

        await asyncio.gather(*post_refute_tasks, return_exceptions=True)
        await asyncio.gather(*dislike_tasks, return_exceptions=True)


        print("Refute/Inoculate phase finished.")
        return "silent"

    # @listen("silent")
    # # @persist(CustomSQLitePersistence(db_path="rumor_flow_data.sqlite"))
    # async def silent(self):
    #     print("Flow is silent, no rumors detected or triggered.")
    #     import time
    #     time.sleep(5)
    #     return

    # Helper for environment actions with error handling

    async def perform_env_action(self, agent: SocialAgent, action_type: ActionType, args: dict, target_user_id: Optional[int] = None) -> Optional[dict]:
        """ Safely performs an environment action for an agent. """
        action_name = action_type.value
        log_target = f" for target context {target_user_id}" if target_user_id else ""
        try:
            action_func = getattr(agent.env.action, action_name)
            result = await action_func(**args)
            print(
                f"Agent {agent.agent_id} performed action {action_name}{log_target}. Result: {result}")
            return result
        except AttributeError:
            print(
                f"Error: Action '{action_name}' not found for agent {agent.agent_id}.")
            return None
        except Exception as e:
            print(
                f"Error performing action {action_name} for agent {agent.agent_id}{log_target}: {e}")
            return None
