#!/usr/bin/env python
import sys
import warnings
import asyncio
from typing import List
import json
from pydantic import BaseModel
from datetime import datetime

from oasis.social_agent import AgentGraph, SocialAgent
from oasis.social_platform.typing import ActionType

from crewai.flow.flow import Flow, FlowState, listen, or_, and_, router, start
from crewai.flow.persistence import persist
from crewai import LLM

from rumor_control.src.rumor_control_group.crews.rumor_identify_crew import RumorIdentifyCrew
from rumor_control.src.rumor_control_group.crews.susceptibility_test_crew import SusceptibilityTestCrew
from rumor_control.src.rumor_control_group.crews.recommend_predict_crew import recommend_predict_crew
from rumor_control.src.rumor_control_group.crews.rumor_refute_crew import rumor_refute_crew
from rumor_control.src.rumor_control_group.crews.rumor_inoculation_crew import rumor_inoculation_crew
from rumor_control.src.rumor_control_group.crews.broadcast_crew import broadcast_crew
# from rumor_control.src.rumor_control_group.crews.recommend_predict_crew import recommend_predict_crew

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

# This main file is intended to be a way for you to run your
# crew locally, so refrain from adding unnecessary logic into this file.
# Replace with inputs you want to test with, it will automatically
# interpolate any tasks and agents information

class UserState(BaseModel):
    susceptible: set[int] = set()
    infected: set[int] = set()
    recovered: set[int] = set()
    still_infected: set[int] = set()

class InspectState(BaseModel):
    private_territory: set[int] = set()
    userstate: UserState = UserState()
    triggered: bool = False
    
class RumorSource(BaseModel):
    # created_time: datetime
    post_id : int
    user_id: int
    content: str = ""
    refute: str = ""
    topic: str = ""

class RumorInfectionState(FlowState):
    class Config:
        arbitrary_types_allowed = True
    agent_graph: AgentGraph = AgentGraph()
    anchor_users: List[int] = [3,7,34,9,55]
    inspect_division: dict[int, InspectState] = {}
    rumor_sources: List[RumorSource] = []
    territory: set[int] = set(anchor_users) #所有已经纳入监控的结点，防止领地重叠
    refute_applyment: set[int] = set()
    inoculation_applyment: set[int] = set()
    newly_infected: tuple[int] = []
    broadcast_anchors: dict[int, str] = {}


#类初始化输入：private_territory,anchor_users
@persist
class RumorControlFlow(Flow[RumorInfectionState]):
    initial_state = RumorInfectionState
    
    def __init__(self, private_territory: List[int], anchor_users: List[int]):
        for i in range(len(anchor_users)):
            self.state.inspect_division[anchor_users[i]].private_territory.add(private_territory[i])
        # self.llm = LLM(model="glm-4", response_format=str)

    @start()
    def detect(self, agent_graph):
        self.state.agent_graph = agent_graph
        self.state.newly_infected = tuple()
        # new_rumor = []
        self.state.refute_applyment = set()
        self.state.inoculation_applyment = set()
        
        anchor_users = agent_graph.get_agents(self.state.anchor_users)
        for anchor_user in anchor_users:
            anchor_id = anchor_user.agent_id
            print(f"anchor_user_{anchor_id} receive message:")
            context = anchor_user.env.to_text_prompt()
            posts = json.dumps(context[2]["posts"][0], indent=4)
            #对一个锚点用户本轮收到的所有帖子进行遍历
            for post in posts:
                print("post: ",post)
                user_id = int(post["user_id"])
                def is_rumor(post):
                    if post["content"] in [r.content for r in self.state.rumor_sources]:
                        x = "old"
                    elif RumorIdentifyCrew().crew().kickoff(input=post["content"]) == "yes":
                        x = "new"
                    else: 
                        x = "not_rumor"
                    return x
                
                #来自广域网的消息 TODO :告知队友
                if user_id not in self.state.inspect_division[anchor_id].private_territory:
                    pass
                
                x = is_rumor(post)
                print(x)
                if not self.state.inspect_division[anchor_id].triggered:
                    if x != "not_rumor":#首次监听到谣言
                        self.state.inspect_division[anchor_id].triggered = True
                        print(f"anchor_user_{anchor_id} triggered! -------------------------")
                    else: #未监听到谣言
                        continue
                #易感者发帖
                if user_id in self.state.inspect_division[anchor_id].userstate.susceptible:
                    # 如果收到来自该用户的谣言，默认该用户已经被感染
                    if x != "not_rumor":
                        if x == "new":
                            # new_rumor.append(RumorSource(post["post_id"], user_id, post["content"],""))
                            response = self.llm.call(
                                f"you see a post: {post}. Categorize the topic of the post. don't output anything else."
                            )
                            topic = response.choices[0].message.content
                            print("rumor topic: ",topic)
                            self.state.rumor_sources.append(RumorSource(post["post_id"], user_id, post["content"],"",topic))
                        self.state.inspect_division[anchor_id].userstate.susceptible.remove(user_id)
                        self.state.inspect_division[anchor_id].userstate.infected.add(user_id)
                        #其所有关注者也感染
                        g = self.state.agent_graph.graph
                        incoming_edges = g.es.select(_target=user_id)
                        followers = [edge.source for edge in incoming_edges]
                        self.state.inspect_division[anchor_id].userstate.infected.update(set(followers))
                        #若已经处理过则不再处理
                        self.state.newly_infected = (list(set(followers) - self.state.territory), user_id) 
                        self.state.refute_applyment.add(user_id)
                    #如果易感者发帖不再涉及谣言，则视为康复
                    else:
                        self.state.inspect_division[anchor_id].userstate.infected.remove(user_id)
                        self.state.inspect_division[anchor_id].userstate.still_infected.add(user_id)
                        
                #感染者发帖
                elif user_id in self.state.inspect_division[anchor_id].userstate.infected:
                    if x == "not_rumor": 
                        self.state.inspect_division[anchor_id].userstate.infected.remove(user_id)
                        self.state.inspect_division[anchor_id].userstate.recovered.add(user_id)
                    #如果继续转发，则视为无药可救
                    else:
                        self.state.inspect_division[anchor_id].userstate.infected.remove(user_id)
                        self.state.inspect_division[anchor_id].userstate.still_infected.add(user_id)
                #TODO 未能治愈
                elif user_id in self.state.inspect_division[anchor_id].userstate.still_infected:
                    if x == "not_rumor":
                        self.state.inspect_division[anchor_id].userstate.still_infected.remove(user_id)
                        self.state.inspect_division[anchor_id].userstate.recovered.add(user_id)
                #TODO: 鼓励康复者辟谣
                elif user_id in self.state.inspect_division[anchor_id].userstate.recovered:
                    pass
            
            #TODO: 局势分析crew
        
    @router(detect)
    def if_triggered(self):
        for anc in self.state.anchor_users:
            if self.state.inspect_division[anc].triggered:
                print("detect finished, ready to select. ")
                return "activated"
        print("no rumor detected. ")
        return "silent"
    
    @listen("activated")
    def select(self):
        newly_infected = self.state.newly_infected
        for anchor_user in self.state.agent_graph.get_agents(self.state.anchor_users):
            anchor_id = anchor_user.agent_id
            if not self.state.inspect_division[anchor_id].triggered:
                continue
            #对于所有感染用户，选择最易感的20%辟谣
            print("selecting refute users ...")
            ref_apply = SusceptibilityTestCrew().crew().kickoff(
                input={
                    "anchor_user":anchor_user,
                    "candidates":newly_infected[0],
                    "graph":self.state.agent_graph,
                    "proportion":0.2
                    }#返回易感用户列表
                )
            self.state.refute_applyment.update(set(ref_apply))
            self.state.territory.update(set(ref_apply+[newly_infected[1]])) #更新领土(已经采取行动)
            self.state.inspect_division[anchor_id].private_territory.update(set(ref_apply+[newly_infected[1]]))
            
            g = self.state.agent_graph.graph
            #选择预接种用户
            print("selecting inoculation users ...")
            for user in self.state.agent_graph.get_agents(newly_infected[0]):
                #获取感染用户的所有关注者
                incoming_edges = g.es.select(_target=user.agent_id)
                sus = list(set([edge.source for edge in incoming_edges]) - self.state.territory)
            inoc_apply = SusceptibilityTestCrew().crew().kickoff(
                input={
                    "anchor_user":anchor_user,
                    "candidates":sus,
                    "graph":self.state.agent_graph,
                    "proportion":0.2
                    }#返回预接种用户列表
                )
            self.state.inoculation_applyment.update(set(inoc_apply))
            self.state.territory.update(set(inoc_apply)) #更新领土(正在监视)
            self.state.inspect_division[anchor_id].private_territory.update(set(inoc_apply))
            
            #通过推荐预测实现跨域预接种
            print("selecting cross-domain inoculation users ...")
            candidates = []
            map = {}
            n = self.state.anchor_users
            for other_anchor in n.remove(anchor_user):
                #如果还未传播到该锚点或传播范围较小
                if list.count(self.state.inspect_division[other_anchor.agnet_id].userstate.susceptible) < 2:#<min
                    cand = self.state.inspect_division[other_anchor.agnet_id].userstate.susceptible[0]
                    candidates.append(cand)
                    map[cand] = other_anchor
            
            rec_apply = recommend_predict_crew.crew().kickoff(
                input={
                    "user":candidates,
                    "graph":self.state.agent_graph,
                    }
                )
            self.state.inoculation_applyment.update(set(rec_apply))
            self.state.territory.update(set(rec_apply))
            for rec_cand in rec_apply:
                self.state.inspect_division[map[rec_cand]].private_territory.add(rec_cand)
                #激活该锚点使之开始监听
                self.state.inspect_division[map[rec_cand]].triggered = True
        
        print("select finished, ready to refute. ")
    
    @listen("silent")
    def silent(self):
        pass
        
    @listen(select)
    def refute(self):
        #每个锚点在自己的领地内行动
        for anchor_user in self.state.agent_graph.get_agents(self.state.anchor_users):
            anchor_id = anchor_user.agent_id
            if not self.state.inspect_division[anchor_id].triggered:
                continue
            #发私信前先关注
            follows = (self.state.inspect_division[anchor_user].private_territory).intersection(self.state.refute_applyment.union(self.state.inoculation_applyment))
            for followee in follows:
                args = {"followee_id": followee}
                # TODO: Implement follow action
                result = getattr(self.env.action, ActionType.FOLLOW.value)(**args)
            #辟谣
            private_refute = list((self.state.refute_applyment).intersection(self.state.inspect_division[anchor_user].private_territory))
            if self.state.rumor_sources[0].refute =="":
                refute_text = rumor_refute_crew().crew().kickoff(
                        input={
                            "user":infected_user,
                            "rumor":self.state.rumor_sources[0].content,
                            "graph":self.state.agent_graph,
                            }
                        )
            else:
                refute_text = self.state.rumor_sources[0].refute
            for infected_user in self.state.agent_graph.get_agents(private_refute):
                # TODO: Implement refute
                args = {"content": refute_text}
                refute_result = getattr(anchor_user.env.action, ActionType.CREATE_POST.value)(**args)
                infected_user.private_message_id = refute_result["post_id"]
                
            #定制化预接种
            for sus_user in self.state.agent_graph.get_agents(self.state.inoculation_applyment):
                gen_inoculation_text = rumor_inoculation_crew().crew().kickoff(
                    input={
                        "user":sus_user,
                        "rumor":self.state.rumor_sources[0].content,
                        "graph":self.state.agent_graph,
                        }
                    )
                args = {"content": gen_inoculation_text}
                inoc_result = getattr(anchor_user.env.action, ActionType.CREATE_POST.value)(**args)
                sus_user.private_message_id = inoc_result["post_id"]
            
            #接种广谱疫苗 TODO: broadcast crew
            if anchor_id in self.state.broadcast_anchors.keys():
                gen_broadcast_text = broadcast_crew().crew().kickoff(
                        input={
                            "user":anchor_id,
                            "rumor":self.state.rumor_sources[0].content,
                            "graph":self.state.agent_graph,
                            }
                        )
                args = {"infected_user": sus_user, "refute_text": gen_inoculation_text}
                result = getattr(anchor_user.env.action,ActionType.CREATE_POST.value)(**args)
                post_id = result["post_id"]
                #发动群众点赞评论
                n = self.state.anchor_users.remove(anchor_id)
                for anchor in n:
                    args = {"post_id": post_id}
                    getattr(self.state.agent_graph.get_agent(anchor), ActionType.LIKE_POST.value)(**args)
        
        print("refute finished, ready for next round. ")    
        

# def kickoff():
#     """
#     Run the flow.
#     """
#     lead_score_flow = LeadScoreFlow()
#     lead_score_flow.kickoff()


def plot():
    """
    Plot the flow.
    """
    lead_score_flow = RumorControlFlow()
    lead_score_flow.plot()


if __name__ == "__main__":
    plot()
#     kickoff()


# def train():
#     """
#     Train the crew for a given number of iterations.
#     """
#     inputs = {
#         "topic": "AI LLMs",
#         'current_year': str(datetime.now().year)
#     }
#     try:
#         RumorControlGroup().crew().train(n_iterations=int(sys.argv[1]), filename=sys.argv[2], inputs=inputs)

#     except Exception as e:
#         raise Exception(f"An error occurred while training the crew: {e}")

# def replay():
#     """
#     Replay the crew execution from a specific task.
#     """
#     try:
#         RumorControlGroup().crew().replay(task_id=sys.argv[1])

#     except Exception as e:
#         raise Exception(f"An error occurred while replaying the crew: {e}")

# def test():
#     """
#     Test the crew execution and returns the results.
#     """
#     inputs = {
#         "topic": "AI LLMs",
#         "current_year": str(datetime.now().year)
#     }
    
#     try:
#         RumorControlGroup().crew().test(n_iterations=int(sys.argv[1]), eval_llm=sys.argv[2], inputs=inputs)

#     except Exception as e:
#         raise Exception(f"An error occurred while testing the crew: {e}")