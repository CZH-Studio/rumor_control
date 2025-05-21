from pydantic import BaseModel
from crewai.flow.flow import FlowState
from oasis.social_agent import AgentGraph
from typing import List, Optional, Any # Added Any
from datetime import datetime

# --- State Definitions (Keep as is) ---
class UserState(BaseModel):
    susceptible: set[int] = set()
    infected: set[int] = set()
    recovered: set[int] = set()
    still_infected: set[int] = set()
    class Config:
        # 当导出为 JSON 时，把 set 转成 list
        json_encoders = { set: list }

class InspectState(BaseModel):
    private_territory: set[int] = set()
    userstate: UserState = UserState()
    triggered: bool = False
    class Config:
        json_encoders = { set: list }

class RumorSource(BaseModel):
    # created_time: datetime
    post_id : int
    user_id: int
    content: str = ""
    refute: str = ""
    topic: str = ""
    created_timestep: int

class RumorInfectionState(FlowState):
    class Config:
        arbitrary_types_allowed = True
        json_encoders = { set: list }
    # Ensure AgentGraph default is handled if needed, or passed in kickoff_async
    # agent_graph: Optional[AgentGraph] = None # Make optional if passed via kickoff
    anchor_users: List[int] = []
    inspect_division: dict[int, InspectState] = {}
    rumor_sources: List[RumorSource] = []
    territory: set[int] = set() # Initialize territory based on anchor_users later
    refute_applyment: set[int] = set()
    inoculation_applyment: set[int] = set()
    # newly_infected: dict[int,tuple] = [] # Initialize as empty list
    broadcast_anchors: dict[int, str] = {}
    post_refute: set[tuple[int, int, str, str]] = set()

# class RumorInfectionState(FlowState): # 假设 RumorInfectionState 继承自 FlowState
#     # ... 其他状态属性 ...
#     # agent_graph: Optional[AgentGraph] = None # 移除这行，或者确保它不被直接赋值
#     anchor_users: List[int] = Field(default_factory=list)
#     territory: Set[int] = Field(default_factory=set)
#     inspect_division: Dict[int, InspectState] = Field(default_factory=dict)
#     newly_infected: Tuple = Field(default_factory=tuple)
#     refute_applyment: Set[int] = Field(default_factory=set)
#     inoculation_applyment: Set[int] = Field(default_factory=set)
#     rumor_sources: List[RumorSource] = Field(default_factory=list)
#     broadcast_anchors: Dict[int, Any] = Field(default_factory=dict) # 添加 Any 类型或具体类型
#     # 你可能需要传递 agent_graph 的序列化表示或ID，然后在需要时重新获取
#     # agent_graph_data: Optional[Dict[str, Any]] = None # 例如，存储序列化的图数据

#     # ... (其他方法)