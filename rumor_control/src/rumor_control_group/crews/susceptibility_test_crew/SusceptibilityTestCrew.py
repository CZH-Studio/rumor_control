from typing import List

from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.knowledge.source.text_file_knowledge_source import TextFileKnowledgeSource

textknowledge = TextFileKnowledgeSource(file_paths=["susceptable_features.txt"])

@CrewBase
class SusceptibilityTestCrew:
    """Susceptibility Test Crew"""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def suscept_test_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["suscept_test_agent"],
            llm = "glm-4-flash",
            verbose=True,
        )

    @task
    def suscept_evaluation_task(self) -> Task:
        return Task(
            config=self.tasks_config["suscept_evaluation"],
            output_pydantic=List[dict[str, int|str]],
        )

    @crew
    def crew(self) -> Crew:
        """Creates the Lead Score Crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            llm="glm-4",
            memory=True,
            process=Process.sequential,
            verbose=True,
            knowledge_sources=[textknowledge],
        )
