from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.knowledge.source.text_file_knowledge_source import TextFileKnowledgeSource
from typing import List

textknowledge = TextFileKnowledgeSource(file_paths=["recommendation_systems.txt"])

@CrewBase
class RecommendPredictCrew:
    """recommend predict crew"""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def recommend_predict_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["recommend_predict_agent"],
            verbose=True,
            llm="glm-4",
        )

    @task
    def recsys_predict_task(self) -> Task:
        return Task(
            config=self.tasks_config["recsys_predict"],
            output_pydantic=List[int],
        )

    @crew
    def crew(self) -> Crew:
        """Creates the recommend predict crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
            llm="glm-4",
            knowledge_sources=[textknowledge],
        )
