#!/usr/bin/env python
import sys
import warnings
import asyncio
from typing import List
import json
from oasis.social_agent import AgentGraph, SocialAgent
from crewai.flow.flow import Flow, listen, or_, router, start, persist
from pydantic import BaseModel

from datetime import datetime

from rumor_control_group.crews.rumor_identify_crew import rumor_identify_crew

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

# This main file is intended to be a way for you to run your
# crew locally, so refrain from adding unnecessary logic into this file.
# Replace with inputs you want to test with, it will automatically
# interpolate any tasks and agents information

class UserState(BaseModel):
    susceptible: list[int]
    infected: list[int]
    recovered: list[int]
    still_infected: list[int]

class InspectState(BaseModel):
    inspect_nodes: List[int]
    userstate: UserState
    
class RumorSource(BaseModel):
    created_time: datetime
    post_id : int
    user_id: int
    content: str
    refute: str

class RumorInfectionState(BaseModel):
    agent_graph: AgentGraph
    anchor_users: List[int]
    inspect_division: dict[int, InspectState]
    inspect_nodes: List[int]
    rumor_sources: List[RumorSource]
    refute_applyment: List[int]
    inoculation_applyment: List[int]
    

@persist
class RumorControlFlow(Flow[RumorInfectionState]):
    initial_state = RumorInfectionState
    def __init__(self):
        self.state.anchor_users = [3,7,34,9,55]

    @start()
    def detect(self, agent_graph):
        self.state.agent_graph = agent_graph
        anchor_users = agent_graph.get_agents(self.state.anchor_users)
        print("anchor_users: ",anchor_users)
        for anchor_user in anchor_users:
            context = anchor_user.env.to_text_prompt()
            posts = json.dumps(context[2]["posts"][0], indent=4)
            for post in posts:
                print("post: ",post)
                user_id = int(post["user_id"])
                #来自广域网的消息
                if user_id not in self.state.inspect_division[anchor_user.agnet_id].inspect_nodes:
                    pass
                elif user_id in self.state.inspect_division[anchor_user.agnet_id].userstate.susceptible:
                    for source in self.state.rumor_sources:
                        if post["content"] == source.content or rumor_identify_crew().crew().kickoff(input=post["content"] == "yes"): #转发谣言
                            self.state.inspect_division[anchor_user.agnet_id].userstate.susceptible.remove(user_id)
                            self.state.inspect_division[anchor_user.agnet_id].userstate.infected.append(user_id)
                            self.state.refute_applyment.append(user_id)
                            break
                elif user_id in self.state.inspect_division[anchor_user.agnet_id].userstate.infected:
                    flag = False
                    for source in self.state.rumor_sources:
                        if not post["content"] == source.content and not rumor_identify_crew().crew().kickoff(input=post["content"] == "yes"): #不再转发谣言
                            self.state.inspect_division[anchor_user.agnet_id].userstate.infected.remove(user_id)
                            self.state.inspect_division[anchor_user.agnet_id].userstate.recovered.append(user_id)
                            flag = True
                            break
                    if not flag:
                        self.state.inspect_division[anchor_user.agnet_id].userstate.infected.remove(user_id)
                        self.state.inspect_division[anchor_user.agnet_id].userstate.still_infected.append(user_id)
                elif user_id in self.state.inspect_division[anchor_user.agnet_id].userstate.still_infected:
                    pass
        return "detect finished, ready to refute. "
          

#     @listen(detect)
#     def select(self):
        
        
#     @listen(select)
#     def refute(self):
        
        
        

#     @router(score_leads)
#     def human_in_the_loop(self):
#         print("Finding the top 3 candidates for human to review")

#         # Combine candidates with their scores using the helper function
#         self.state.hydrated_candidates = combine_candidates_with_scores(
#             self.state.candidates, self.state.candidate_score
#         )

#         # Sort the scored candidates by their score in descending order
#         sorted_candidates = sorted(
#             self.state.hydrated_candidates, key=lambda c: c.score, reverse=True
#         )
#         self.state.hydrated_candidates = sorted_candidates

#         # Select the top 3 candidates
#         top_candidates = sorted_candidates[:3]

#         print("Here are the top 3 candidates:")
#         for candidate in top_candidates:
#             print(
#                 f"ID: {candidate.id}, Name: {candidate.name}, Score: {candidate.score}, Reason: {candidate.reason}"
#             )

#         # Present options to the user
#         print("\nPlease choose an option:")
#         print("1. Quit")
#         print("2. Redo lead scoring with additional feedback")
#         print("3. Proceed with writing emails to all leads")

#         choice = input("Enter the number of your choice: ")

#         if choice == "1":
#             print("Exiting the program.")
#             exit()
#         elif choice == "2":
#             feedback = input(
#                 "\nPlease provide additional feedback on what you're looking for in candidates:\n"
#             )
#             self.state.scored_leads_feedback = feedback
#             print("\nRe-running lead scoring with your feedback...")
#             return "scored_leads_feedback"
#         elif choice == "3":
#             print("\nProceeding to write emails to all leads.")
#             return "generate_emails"
#         else:
#             print("\nInvalid choice. Please try again.")
#             return "human_in_the_loop"

#     @listen("generate_emails")
#     async def write_and_save_emails(self):
#         import re
#         from pathlib import Path

#         print("Writing and saving emails for all leads.")

#         # Determine the top 3 candidates to proceed with
#         top_candidate_ids = {
#             candidate.id for candidate in self.state.hydrated_candidates[:3]
#         }

#         tasks = []

#         # Create the directory 'email_responses' if it doesn't exist
#         output_dir = Path(__file__).parent / "email_responses"
#         print("output_dir:", output_dir)
#         output_dir.mkdir(parents=True, exist_ok=True)

#         async def write_email(candidate):
#             # Check if the candidate is among the top 3
#             proceed_with_candidate = candidate.id in top_candidate_ids

#             # Kick off the LeadResponseCrew for each candidate
#             result = await (
#                 LeadResponseCrew()
#                 .crew()
#                 .kickoff_async(
#                     inputs={
#                         "candidate_id": candidate.id,
#                         "name": candidate.name,
#                         "bio": candidate.bio,
#                         "proceed_with_candidate": proceed_with_candidate,
#                     }
#                 )
#             )

#             # Sanitize the candidate's name to create a valid filename
#             safe_name = re.sub(r"[^a-zA-Z0-9_\- ]", "", candidate.name)
#             filename = f"{safe_name}.txt"
#             print("Filename:", filename)

#             # Write the email content to a text file
#             file_path = output_dir / filename
#             with open(file_path, "w", encoding="utf-8") as f:
#                 f.write(result.raw)

#             # Return a message indicating the email was saved
#             return f"Email saved for {candidate.name} as {filename}"

#         # Create tasks for all candidates
#         for candidate in self.state.hydrated_candidates:
#             task = asyncio.create_task(write_email(candidate))
#             tasks.append(task)

#         # Run all email-writing tasks concurrently and collect results
#         email_results = await asyncio.gather(*tasks)

#         # After all emails have been generated and saved
#         print("\nAll emails have been written and saved to 'email_responses' folder.")
#         for message in email_results:
#             print(message)


# def kickoff():
#     """
#     Run the flow.
#     """
#     lead_score_flow = LeadScoreFlow()
#     lead_score_flow.kickoff()


# def plot():
#     """
#     Plot the flow.
#     """
#     lead_score_flow = LeadScoreFlow()
#     lead_score_flow.plot()


# if __name__ == "__main__":
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
