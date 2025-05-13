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
    # Ensure AgentGraph default is handled if needed, or passed in kickoff_async
    agent_graph: Optional[AgentGraph] = None # Make optional if passed via kickoff
    anchor_users: List[int] = []
    inspect_division: dict[int, InspectState] = {}
    rumor_sources: List[RumorSource] = []
    territory: set[int] = set() # Initialize territory based on anchor_users later
    refute_applyment: set[int] = set()
    inoculation_applyment: set[int] = set()
    newly_infected: tuple = () # Initialize as empty tuple
    broadcast_anchors: dict[int, str] = {}