#!/usr/bin/env python
import sys
import random
import warnings
import asyncio
from typing import List, Optional, Any # Added Any
import json

from oasis.social_agent import AgentGraph, SocialAgent
from oasis.social_platform.typing import ActionType

from crewai.flow.flow import Flow, listen, router, start, FlowPersistence
from crewai.flow.persistence import persist, SQLiteFlowPersistence # Added SQLiteFlowPersistence for default
from crewai import LLM

from rumor_control.src.rumor_control_group.crews.rumor_identify_crew import RumorIdentifyCrew
from rumor_control.src.rumor_control_group.crews.susceptibility_test_crew import SusceptibilityTestCrew
from rumor_control.src.rumor_control_group.crews.recommend_predict_crew import RecommendPredictCrew
from rumor_control.src.rumor_control_group.crews.rumor_refute_crew import rumor_refute_crew
from rumor_control.src.rumor_control_group.crews.rumor_inoculation_crew import rumor_inoculation_crew
from rumor_control.src.rumor_control_group.crews.broadcast_crew import broadcast_crew

from rumor_control.src.rumor_control_group.data_type.data_types import *
from rumor_control.src.rumor_control_group.data_type.constants import *

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

# Use method-level persistence
class RumorControlFlow(Flow[RumorInfectionState]):
    initial_state = RumorInfectionState

    # __init__ remains synchronous
    def __init__(
        self,
        private_territory: List[int],
        anchor_users: List[int],
        specialized_refute: bool,
        persistence: Optional[FlowPersistence] = None, # Optional persistence
        **kwargs: Any # Accept other kwargs for Flow base class
    ):
        # Call super().__init__ first is good practice
        super().__init__(persistence=persistence, **kwargs)

        # Initialize state attributes after super init
        # self.state comes from Flow base or initial_state instantiation
        if self._state is None: # Ensure state exists
             self._state = self.initial_state() if isinstance(self.initial_state, type) else self.initial_state

        self.state.anchor_users = anchor_users
        self.state.territory = set(anchor_users) # Initialize territory 
        self.specialized_refute = specialized_refute

        # Initialize inspect_division properly
        if not hasattr(self.state, 'inspect_division') or self.state.inspect_division is None:
             self.state.inspect_division = {}

        for i in range(len(anchor_users)):
            anchor_id = anchor_users[i]
            if anchor_id not in self.state.inspect_division:
                self.state.inspect_division[anchor_id] = InspectState() # Create state if needed
            # Ensure private_territory is a set
            if not isinstance(self.state.inspect_division[anchor_id].private_territory, set):
                self.state.inspect_division[anchor_id].private_territory = set()
            # Handle potential index error if private_territory list is shorter
            if i < len(private_territory):
                self.state.inspect_division[anchor_id].private_territory.add(private_territory[i])

        self.llm = LLM(model="glm-4")

    # --- Flow Methods (Make Async) ---

    @start()
    @persist()
    async def detect(self):
        agent_graph = self.state.agent_graph
        if agent_graph is None:
            print("Error: agent_graph not found in flow state during detect.")
            return

        self.state.newly_infected = tuple()
        self.state.refute_applyment = set()
        self.state.inoculation_applyment = set()

        anchor_users = agent_graph.get_agents(self.state.anchor_users)
        if not anchor_users:
             print("Warning: No anchor users found in the graph.")
             return

        for anchor_user in anchor_users:
            anchor_id = anchor_user.agent_id
            print(f"anchor_user_{anchor_id} receive message:")
            # Assuming env.to_text_prompt is synchronous or fast
            context_data = anchor_user.env.get_posts_list()

            for post in context_data:
                if not isinstance(post, dict): # Ensure post is a dict
                    print(f"Skipping invalid post item for anchor_user_{anchor_id}: {post}")
                    continue

                print("post: ", json.dumps(post, indent=2))
                user_id = int(post.get("user_id"))
                content = post.get("content", "")
                post_id = post.get("post_id")

                # --- is_rumor (can remain sync helper or integrate) ---
                def is_rumor_sync(post_content):
                    if post_content in [r.content for r in self.state.rumor_sources]:
                        return "old"
                    try:
                         # Ensure input is just the content string
                         crew_input = {"input": post_content}
                         print(f"Running RumorIdentifyCrew with input: {crew_input}")
                         result = RumorIdentifyCrew().crew().kickoff(inputs=crew_input)
                         print(f"RumorIdentifyCrew result: {result}")
                         if isinstance(result, str) and result.strip().lower() == "yes":
                             return "new"
                         else:
                             return "not_rumor"
                    except Exception as e:
                        print(f"Error during RumorIdentifyCrew kickoff: {e}")
                        return "not_rumor" # Default to not_rumor on error

                #来自广域网的消息 TODO :告知队友
                if user_id not in self.state.inspect_division[anchor_id].private_territory:
                    pass

                # Call the sync helper
                rumor_status = is_rumor_sync(content)
                print(f"Rumor status for post {post_id}: {rumor_status}")

                # Initialize UserState if necessary
                if not isinstance(self.state.inspect_division[anchor_id].userstate, UserState):
                     self.state.inspect_division[anchor_id].userstate = UserState()
                current_user_state = self.state.inspect_division[anchor_id].userstate

                if not self.state.inspect_division[anchor_id].triggered:
                    if rumor_status != "not_rumor": #首次监听到谣言
                        self.state.inspect_division[anchor_id].triggered = True
                        print(f"anchor_user_{anchor_id} triggered! -------------------------")
                    else: #未监听到谣言
                        continue # Skip rest of processing if not triggered and not a rumor

                # Ensure sets exist before modifying
                current_user_state.susceptible = current_user_state.susceptible or set()
                current_user_state.infected = current_user_state.infected or set()
                current_user_state.recovered = current_user_state.recovered or set()
                current_user_state.still_infected = current_user_state.still_infected or set()


                #易感者发帖
                if user_id in current_user_state.susceptible:
                    if rumor_status != "not_rumor":
                        if rumor_status == "new":
                            # Using await for potential async LLM call
                            llm_call_needed = True # Set flag if LLM needed
                            if llm_call_needed and hasattr(self, 'llm') and self.llm:
                                 try:
                                     response = await self.llm.acall(
                                         f"you see a post: {json.dumps(post)}. Categorize the topic of the post. don't output anything else."
                                     )
                                     topic = response.choices[0].message.content
                                     print("rumor topic: ",topic)
                                 except Exception as e:
                                     print(f"LLM call failed: {e}")
                                     topic = "news"
                            else:
                                # Fallback or skip topic generation if LLM not available/needed
                                print("Skipping LLM topic generation.")
                                topic = "news" # Assign default topic
                            self.state.rumor_sources.append(RumorSource(post_id=post_id, user_id=user_id, content=content, refute="", topic=topic))

                        current_user_state.susceptible.discard(user_id)
                        current_user_state.infected.add(user_id)

                        #其所有关注者也感染
                        g = agent_graph.graph
                        try:
                            # Ensure user_id exists as a vertex
                            user_vertex = g.vs.find(name=str(user_id)) # Assuming node names are strings
                            # Find followers (edges pointing TO user_vertex)
                            incoming_edges = g.es.select(_target=user_vertex.index)
                            followers = [int(g.vs[edge.source]["name"]) for edge in incoming_edges] # Convert name back to int
                            # Update sets safely
                            newly_infected_followers = set(followers) - current_user_state.infected - current_user_state.recovered - current_user_state.still_infected
                            current_user_state.infected.update(newly_infected_followers)

                            # Newly infected = followers not already in territory + the user themselves
                            # Ensure self.state.territory is a set
                            if not isinstance(self.state.territory, set): self.state.territory = set()
                            new_infections_for_territory = (newly_infected_followers - self.state.territory)
                            self.state.newly_infected = (list(new_infections_for_territory), user_id)
                            print(f"Newly infected: {self.state.newly_infected}")

                            # Add user to refute list
                            self.state.refute_applyment.add(user_id)
                        except ValueError:
                            print(f"User ID {user_id} not found in graph vertices.")
                        except Exception as e:
                            print(f"Error processing followers for user {user_id}: {e}")

                    else: #如果易感者发帖不再涉及谣言，则视为康复
                        current_user_state.susceptible.discard(user_id)
                        current_user_state.recovered.add(user_id)

                #感染者发帖
                elif user_id in current_user_state.infected:
                    if rumor_status == "not_rumor":
                        current_user_state.infected.discard(user_id)
                        current_user_state.recovered.add(user_id)
                    else: #如果继续转发谣言
                        current_user_state.infected.discard(user_id)
                        current_user_state.still_infected.add(user_id) #辟谣次数用光，不再进行辟谣

                #未能治愈者发帖
                elif user_id in current_user_state.still_infected:
                    if rumor_status == "not_rumor":
                        current_user_state.still_infected.discard(user_id)
                        current_user_state.recovered.add(user_id)
                    # else: Still infected, no change

                #康复者发帖
                elif user_id in current_user_state.recovered:
                    # TODO: Could re-infect? Or promote refutation?
                    pass # No action for recovered users posting for now
                
                #TODO: 局势分析crew,用于决定broadcastCrew名单


    @router(detect)
    @persist()
    async def if_triggered(self):
        for anc in self.state.anchor_users:
             # Check if anchor exists and is triggered
             if anc in self.state.inspect_division and self.state.inspect_division[anc].triggered:
                print("detect finished, ready to select. ")
                return "activated"
        print("no rumor detected. ")
        return "silent"
    

    @listen("activated")
    @persist()
    async def select(self):
        agent_graph = self.state.agent_graph # Get graph from state
        if agent_graph is None:
             print("Error: agent_graph not available in select state.")
             return None, None

        newly_infected_tuple = self.state.newly_infected
        # Ensure newly_infected_tuple is a tuple/list with at least one element expected for followers
        if not isinstance(newly_infected_tuple, (tuple, list)) or len(newly_infected_tuple) < 1:
            print("No newly infected users to process in select.")
            return None, None # Or return empty lists: [], []

        # Extract follower list, handle potential errors
        newly_infected_followers = []
        if isinstance(newly_infected_tuple[0], (list, set)):
            newly_infected_followers = list(newly_infected_tuple[0])

        # The user who posted the rumor (might be None if only followers processed)
        infected_poster_id = newly_infected_tuple[1] if len(newly_infected_tuple) > 1 else None

        ref_apply_final = set()
        inoc_apply_final = set()

        processed_anchors = set() # Track anchors whose territory has been processed this round
        
        ref_apply_result_with_reason = {}
        inoc_apply_result_with_reason = {}

        for anchor_user_obj in self.state.agent_graph.get_agents(self.state.anchor_users):
            anchor_id = anchor_user_obj.agent_id

            # Skip if anchor not triggered or already processed its part
            if anchor_id not in self.state.inspect_division or not self.state.inspect_division[anchor_id].triggered or anchor_id in processed_anchors:
                continue
            
            ref_apply_result_with_reason[anchor_id] = {}
            inoc_apply_result_with_reason[anchor_id] = {}

            print(f"Processing selection for anchor {anchor_id}")
            processed_anchors.add(anchor_id) # Mark as processed

            # Combine followers and potentially the original poster for susceptibility test
            users_to_test_ids = set(newly_infected_followers)
            if infected_poster_id:
                users_to_test_ids.add(infected_poster_id)

            # --- Refute Selection ---
            print("selecting refute users ...")
            profiles_for_refute = []
            for user_id in users_to_test_ids:
                agent = self.state.agent_graph.get_agent(user_id)
                if agent and hasattr(agent, 'user_info') and hasattr(agent.user_info, 'description'):
                     profiles_for_refute.append({
                        "user_id": user_id,
                        "user_profile": agent.user_info.description,
                     })
                else:
                    print(f"Warning: Could not get profile for user {user_id}")

            if profiles_for_refute:
                num_to_refute = max(1, int(0.2 * len(profiles_for_refute))) # Refute at least 1 if possible
                try:
                    # NOTE: Crew kickoff is sync. Consider async if available.
                    ref_apply_crew_input = {
                        "num": num_to_refute,
                        "personal_profile": profiles_for_refute,
                    }
                    print(f"Running SusceptibilityTestCrew (refute) with input: {ref_apply_crew_input}")
                    ref_apply_result = SusceptibilityTestCrew().crew().kickoff(inputs=ref_apply_crew_input) # Pass as dict
                    for ref_with_reason in ref_apply_result:
                        ref_apply_result_with_reason[anchor_id][ref_with_reason["user_id"]] = ref_with_reason["reason"]
                    print(f"SusceptibilityTestCrew (refute) result: {ref_apply_result}")
                    

                    if isinstance(ref_apply_result, list):
                         ref = {selec_usr["user_id"] for selec_usr in ref_apply_result if isinstance(selec_usr, dict) and "user_id" in selec_usr}
                         ref_apply_final.update(ref)
                         # Update territory only with users selected for refutation
                         self.state.territory.update(ref)
                         # Also update the specific anchor's private territory
                         self.state.inspect_division[anchor_id].private_territory.update(ref)
                    else:
                        print("Warning: SusceptibilityTestCrew (refute) did not return a list.")

                except Exception as e:
                    print(f"Error during SusceptibilityTestCrew (refute) kickoff: {e}")
            else:
                 print("No profiles available for refutation selection.")


            # --- Inoculation Selection ---
            print("selecting inoculation users ...")
            g = agent_graph.graph
            susceptible_followers_to_inoculate = set()

            # Find followers of the newly infected users (original list + poster)
            for user_id in users_to_test_ids:
                try:
                    user_vertex = g.vs.find(name=str(user_id))
                    # Find followers (edges pointing TO user_vertex)
                    incoming_edges = g.es.select(_target=user_vertex.index)
                    followers_of_infected = {int(g.vs[edge.source]["name"]) for edge in incoming_edges}
                    # Add followers who are not already in territory
                    susceptible_followers_to_inoculate.update(followers_of_infected - self.state.territory)
                except ValueError:
                     print(f"User ID {user_id} not found in graph for inoculation follower search.")
                except Exception as e:
                     print(f"Error finding followers of {user_id} for inoculation: {e}")

            profiles_for_inoculation = []
            for user_id in susceptible_followers_to_inoculate:
                 agent = self.state.agent_graph.get_agent(user_id)
                 if agent and hasattr(agent, 'user_info') and hasattr(agent.user_info, 'description'):
                    profiles_for_inoculation.append({
                        "user_id": user_id,
                        "user_profile": agent.user_info.description,
                    })
                 else:
                    print(f"Warning: Could not get profile for user {user_id} (inoculation)")

            if profiles_for_inoculation:
                num_to_inoculate = max(1, int(0.2 * len(profiles_for_inoculation))) # Inoculate at least 1
                try:
                    inoc_apply_crew_input = {
                        "num": num_to_inoculate,
                        "personal_profile": profiles_for_inoculation,
                    }
                    print(f"Running SusceptibilityTestCrew (inoculation) with input: {inoc_apply_crew_input}")
                    inoc_apply_result = SusceptibilityTestCrew().crew().kickoff(inputs=inoc_apply_crew_input) # Pass as dict
                    for inoc_with_reason in ref_apply_result:
                        inoc_apply_result_with_reason[anchor_id][inoc_with_reason["user_id"]] = inoc_with_reason["reason"]
                    print(f"SusceptibilityTestCrew (inoculation) result: {inoc_apply_result}")

                    if isinstance(inoc_apply_result, list):
                        inoc = {selec_usr["user_id"] for selec_usr in inoc_apply_result if isinstance(selec_usr, dict) and "user_id" in selec_usr}
                        inoc_apply_final.update(inoc)
                        # Update territory
                        self.state.territory.update(inoc)
                        self.state.inspect_division[anchor_id].private_territory.update(inoc)
                    else:
                        print("Warning: SusceptibilityTestCrew (inoculation) did not return a list.")
                except Exception as e:
                    print(f"Error during SusceptibilityTestCrew (inoculation) kickoff: {e}")
            else:
                print("No profiles available for inoculation selection.")


            # --- Cross-Domain Inoculation (Simplified/Corrected Logic) ---
            print("selecting cross-domain inoculation users ...")
            candidates_for_cross_domain = []
            candidate_to_anchor_map = {}

            current_anchor_agent = self.state.agent_graph.get_agent(anchor_id)
            if not current_anchor_agent: continue # Skip if anchor agent not found

            # Find the rumor source related to the current infection event if possible
            # This logic assumes the first rumor source is relevant, might need refinement
            current_rumor = self.state.rumor_sources[0] if self.state.rumor_sources else None
            if not current_rumor:
                print("No rumor source found for cross-domain prediction.")
                continue

            other_anchor_ids = set(self.state.anchor_users) - {anchor_id}

            for other_anchor_id in other_anchor_ids:
                 if other_anchor_id in self.state.inspect_division:
                     other_anchor_state = self.state.inspect_division[other_anchor_id]
                     # Check if the other anchor is NOT triggered (or has very few susceptible - adjust criteria)
                     if not other_anchor_state.triggered:
                          # Select some users from the other anchor's territory as candidates
                          # Using private_territory, limit the number
                          potential_cands = list(other_anchor_state.private_territory - self.state.territory)
                          num_cands_to_take = min(5, len(potential_cands)) # Take up to 5 candidates
                          selected_cands = random.sample(potential_cands, num_cands_to_take) if potential_cands else []

                          for cand_id in selected_cands:
                              candidates_for_cross_domain.append(cand_id)
                              candidate_to_anchor_map[cand_id] = other_anchor_id # Map candidate back to its original anchor
                 else:
                     print(f"No state found for other anchor {other_anchor_id}")


            if candidates_for_cross_domain:
                 receivers_profiles = []
                 for rec_id in candidates_for_cross_domain:
                     rec_agent = self.state.agent_graph.get_agent(rec_id)
                     if rec_agent and hasattr(rec_agent, 'user_info') and hasattr(rec_agent.user_info, 'description'):
                         receivers_profiles.append({"user_id": rec_id, "user_profile": rec_agent.user_info.description})
                     else:
                         print(f"Warning: Could not get profile for cross-domain candidate {rec_id}")

                 if receivers_profiles:
                    num_cross_domain = max(1, int(0.2 * len(receivers_profiles))) # Predict for at least 1
                    poster_agent = self.state.agent_graph.get_agent(current_rumor.user_id) # Agent who posted the rumor
                    poster_profile = poster_agent.user_info.description if poster_agent else "Unknown"

                    try:
                        rec_predict_input = {
                            "num": num_cross_domain,
                            "rec_type": current_anchor_agent.user_info.recsys_type if hasattr(current_anchor_agent.user_info, 'recsys_type') else "default",
                            "poster": {
                                "user_id": current_rumor.user_id,
                                "user_profile": poster_profile,
                                # Pass rumor content or ID - check crew requirements
                                "post": {"content": current_rumor.content, "topic": current_rumor.topic},
                            },
                            "recievers": receivers_profiles, # Correct spelling 'receivers' if crew expects it
                        }
                        print(f"Running RecommendPredictCrew with input: {json.dumps(rec_predict_input, indent=2)}")
                        rec_apply_far_result = RecommendPredictCrew().crew().kickoff(inputs=rec_predict_input) # Pass as dict
                        print(f"RecommendPredictCrew result: {rec_apply_far_result}")

                        if isinstance(rec_apply_far_result, list): # Expecting list of user IDs directly based on previous logic? Adapt if needed.
                            # Assuming result is list of user IDs
                            rec_apply_far_ids = set(rec_apply_far_result)
                            inoc_apply_final.update(rec_apply_far_ids) # Add to final inoculation set
                            self.state.territory.update(rec_apply_far_ids)

                            # Update the territories of the *original* anchors of these candidates
                            for rec_cand_id in rec_apply_far_ids:
                                original_anchor_id = candidate_to_anchor_map.get(rec_cand_id)
                                if original_anchor_id and original_anchor_id in self.state.inspect_division:
                                    self.state.inspect_division[original_anchor_id].private_territory.add(rec_cand_id)
                                    # Activate the other anchor if it wasn't already
                                    if not self.state.inspect_division[original_anchor_id].triggered:
                                        print(f"Activating anchor {original_anchor_id} due to cross-domain inoculation.")
                                        self.state.inspect_division[original_anchor_id].triggered = True
                        else:
                            print("Warning: RecommendPredictCrew did not return a list.")

                    except Exception as e:
                        print(f"Error during RecommendPredictCrew kickoff: {e}")
                 else:
                     print("No receiver profiles for cross-domain prediction.")
            else:
                print("No candidates found for cross-domain inoculation.")


        # Update final sets after processing all anchors
        self.state.refute_applyment = ref_apply_final
        self.state.inoculation_applyment = inoc_apply_final

        print(f"Select finished. Refute Set: {self.state.refute_applyment}, Inoculation Set: {self.state.inoculation_applyment}")
        # Return structure might need adjustment based on how @listen uses it,
        # but returning the final sets seems logical.
        return ref_apply_result_with_reason, inoc_apply_result_with_reason


    @listen("silent")
    async def silent(self):
        print("Flow is silent, no rumors detected or triggered.")
        pass


    @listen(select)
    @persist()
    async def refute(self, ref_apply_result_with_reason, inoc_apply_result_with_reason):
        agent_graph = self.state.agent_graph # Get graph from state
        if agent_graph is None:
             print("Error: agent_graph not available in refute state.")
             return

        print("Refute/Inoculate phase started...")
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
                     "profile": REPRESENTATIVE_USER_PROFILE, # If needed
                     "reason_been_chosen": REPRESENTATIVE_REASON_BEEN_CHOSEN,
                     "rumor": rumor_to_address.content,
                     "topic": rumor_to_address.topic,
                 }
                 print(f"Running rumor_refute_crew with input: {refute_crew_input}")
                 generated_refute = rumor_refute_crew().crew().kickoff(inputs=refute_crew_input) # Pass as dict
                 print(f"rumor_refute_crew result: {generated_refute}")
                 if isinstance(generated_refute, str):
                      refute_text = generated_refute
                      # Optionally store back to rumor_source if desired
                      # rumor_to_address.refute = refute_text
                 else:
                      print("Warning: Refute crew did not return a string. Using default.")
                      refute_text = f"Regarding '{rumor_to_address.content[:50]}...', please verify information before sharing."
             except Exception as e:
                 print(f"Error during refute crew kickoff: {e}")
                 refute_text = f"Regarding '{rumor_to_address.content[:50]}...', please verify information before sharing." # Default on error

        processed_users_for_action = set() # Track users already acted upon this cycle

        # Iterate through anchors to perform actions within their territories
        for anchor_user_obj in self.state.agent_graph.get_agents(self.state.anchor_users):
             anchor_id = anchor_user_obj.agent_id
             if anchor_id not in self.state.inspect_division or not self.state.inspect_division[anchor_id].triggered:
                 continue # Skip inactive anchors

             anchor_territory = self.state.inspect_division[anchor_id].private_territory

             # --- Follow Action (Optional, if needed before messaging) ---
             users_to_potentially_follow = (self.state.refute_applyment | self.state.inoculation_applyment) & anchor_territory
             follow_tasks = []
             for followee_id in users_to_potentially_follow:
                  # Check if already following? Assume env handles this.
                  args = {"followee_id": followee_id}
                  print(f"Anchor {anchor_id} attempting to follow {followee_id}")
                  # Assuming follow action is async
                  follow_tasks.append(getattr(anchor_user_obj.env.action, ActionType.FOLLOW.value)(**args))
             try:
                  await asyncio.gather(*follow_tasks)
                  print(f"Anchor {anchor_id} finished follow attempts.")
             except Exception as e:
                  print(f"Error during follow actions for anchor {anchor_id}: {e}")


             # --- Refutation Actions ---
             users_to_refute_in_territory = (self.state.refute_applyment & anchor_territory) - processed_users_for_action
             refute_tasks = []
             print(f"Anchor {anchor_id} refuting users: {users_to_refute_in_territory}")
             for infected_user_id in users_to_refute_in_territory:
                 infected_user_agent = self.state.agent_graph.get_agent(infected_user_id)
                 if not infected_user_agent: continue
                 if self.sepcialized_refute and infected_user_id in ref_apply_result_with_reason[anchor_id].keys():
                    try:
                        # NOTE: Crew kickoff sync. Generate personalized refute.
                        ref_crew_input = {
                            "profile": sus_user_agent.user_info.description if hasattr(sus_user_agent, 'user_info') else "",
                            "reason_been_chosen": ref_apply_result_with_reason[anchor_id][infected_user_id],
                            "rumor": rumor_to_address.content,
                            "topic": rumor_to_address.topic,
                        }
                        print(f"Running rumor_refute_crew for {sus_user_id}")
                        gen_inoculation_text = rumor_inoculation_crew().crew().kickoff(inputs=ref_crew_input) # Pass as dict
                        print(f"rumor_inoculation_crew result: {gen_inoculation_text}")

                        if not isinstance(gen_inoculation_text, str):
                            print("Warning: Inoculation crew did not return string. Using default.")
                            gen_inoculation_text = f"Be aware of information like '{rumor_to_address.content[:50]}...'. Stay critical!"

                    except Exception as e:
                        print(f"Error during inoculation crew kickoff for {sus_user_id}: {e}")
                        gen_inoculation_text = refute_text

                 else:
                 # Send refute as a post or direct message? Assuming CREATE_POST for broad reach
                    gen_inoculation_text = refute_text
                 args = {"content": gen_inoculation_text}
                 print(f"Anchor {anchor_id} creating refute post targeting user context (indirectly) {infected_user_id}")
                 # Action likely async
                 refute_tasks.append(
                     self.perform_env_action(anchor_user_obj, ActionType.CREATE_POST, args, target_user_id=infected_user_id)
                 )
             results_refute = await asyncio.gather(*refute_tasks, return_exceptions=True)
             
             for infected_user_id in users_to_refute_in_territory:
                 infected_user_agent = self.state.agent_graph.get_agent(infected_user_id)
             infected_user_agent.private_message_id = results_refute["post_id"]
             
             processed_users_for_action.update(users_to_refute_in_territory) # Mark as processed
             # Process results if needed (e.g., store post_id)


             # --- Inoculation Actions ---
             users_to_inoculate_in_territory = (self.state.inoculation_applyment & anchor_territory) - processed_users_for_action
             inoculate_tasks = []
             print(f"Anchor {anchor_id} inoculating users: {users_to_inoculate_in_territory}")
             for sus_user_id in users_to_inoculate_in_territory:
                 sus_user_agent = self.state.agent_graph.get_agent(sus_user_id)
                 if not sus_user_agent: continue

                 try:
                      # NOTE: Crew kickoff sync. Generate personalized inoculation.
                      inoc_crew_input = {
                          "user": {"id": sus_user_id, "profile": sus_user_agent.user_info.description if hasattr(sus_user_agent, 'user_info') else ""},
                          "reason_been_chosen": inoc_apply_result_with_reason[anchor_id][infected_user_id],
                          "rumor": rumor_to_address.content,
                          "topic": rumor_to_address.topic,
                      }
                      print(f"Running rumor_inoculation_crew for {sus_user_id}")
                      gen_inoculation_text = rumor_inoculation_crew().crew().kickoff(inputs=inoc_crew_input) # Pass as dict
                      print(f"rumor_inoculation_crew result: {gen_inoculation_text}")

                      if not isinstance(gen_inoculation_text, str):
                          print("Warning: Inoculation crew did not return string. Using default.")
                          gen_inoculation_text = f"Be aware of information like '{rumor_to_address.content[:50]}...'. Stay critical!"

                 except Exception as e:
                      print(f"Error during inoculation crew kickoff for {sus_user_id}: {e}")
                      gen_inoculation_text = f"Be aware of information like '{rumor_to_address.content[:50]}...'. Stay critical!" # Default

                 # Post inoculation message
                 args = {"content": gen_inoculation_text}
                 print(f"Anchor {anchor_id} creating inoculation post targeting user context (indirectly) {sus_user_id}")
                 # Action likely async
                 inoculate_tasks.append(
                     self.perform_env_action(anchor_user_obj, ActionType.CREATE_POST, args, target_user_id=sus_user_id)
                 )

             results_inoc = await asyncio.gather(*inoculate_tasks, return_exceptions=True)
             
             for sus_user_id in users_to_inoculate_in_territory:
                 sus_user_agent = self.state.agent_graph.get_agent(sus_user_id)
             sus_user_agent.private_message_id = results_inoc["post_id"]
             
             processed_users_for_action.update(users_to_inoculate_in_territory) # Mark as processed
             # Process results if needed

             # --- Broadcast Action (If anchor is designated) ---
             if anchor_id in self.state.broadcast_anchors.keys():
                 print(f"Anchor {anchor_id} performing broadcast...")
                 try:
                     # NOTE: Crew kickoff sync.
                     broadcast_crew_input = {
                         "user": {"id": anchor_id, "profile": anchor_user_obj.user_info.description if hasattr(anchor_user_obj, 'user_info') else ""},
                         "rumor": rumor_to_address.content,
                         # "graph": self.state.agent_graph, # If needed
                     }
                     print("Running broadcast_crew...")
                     gen_broadcast_text = broadcast_crew().crew().kickoff(inputs=broadcast_crew_input) # Pass as dict
                     print(f"broadcast_crew result: {gen_broadcast_text}")

                     if not isinstance(gen_broadcast_text, str):
                         print("Warning: Broadcast crew did not return string. Skipping broadcast.")
                         continue

                 except Exception as e:
                     print(f"Error during broadcast crew kickoff: {e}")
                     continue # Skip broadcast on error

                 args = {"content": gen_broadcast_text}
                 broadcast_post_result = await self.perform_env_action(anchor_user_obj, ActionType.CREATE_POST, args)

                 # If post created successfully, trigger likes from other anchors
                 if broadcast_post_result and isinstance(broadcast_post_result, dict) and "post_id" in broadcast_post_result:
                     post_id = broadcast_post_result["post_id"]
                     print(f"Broadcast post created: {post_id}. Triggering likes...")
                     like_tasks = []
                     other_anchor_ids = set(self.state.anchor_users) - {anchor_id}
                     for other_anchor_id in other_anchor_ids:
                          other_anchor_agent = self.state.agent_graph.get_agent(other_anchor_id)
                          if other_anchor_agent:
                              like_args = {"post_id": post_id}
                              like_tasks.append(
                                  self.perform_env_action(other_anchor_agent, ActionType.LIKE_POST, like_args)
                              )
                     await asyncio.gather(*like_tasks, return_exceptions=True) # Wait for likes

        print("Refute/Inoculate phase finished.")


    # Helper for environment actions with error handling
    async def perform_env_action(self, agent: SocialAgent, action_type: ActionType, args: dict, target_user_id: Optional[int] = None) -> Optional[dict]:
        """ Safely performs an environment action for an agent. """
        action_name = action_type.value
        log_target = f" for target context {target_user_id}" if target_user_id else ""
        try:
            action_func = getattr(agent.env.action, action_name)
            result = await action_func(**args)
            print(f"Agent {agent.agent_id} performed action {action_name}{log_target}. Result: {result}")
            return result
        except AttributeError:
            print(f"Error: Action '{action_name}' not found for agent {agent.agent_id}.")
            return None
        except Exception as e:
            print(f"Error performing action {action_name} for agent {agent.agent_id}{log_target}: {e}")
            return None
