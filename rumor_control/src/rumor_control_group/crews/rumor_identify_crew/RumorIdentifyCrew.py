from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task

@CrewBase
class RumorIdentifyCrew:
    """rumor identify crew"""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def rumor_identifier_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["rumor_identifier_agent"],
            verbose=True,
            allow_delegation=False,
            llm="glm-z1-flash"
        )

    @task
    def rumor_identification_task(self) -> Task:
        return Task(
            config=self.tasks_config["rumor_identification"],
            verbose=True,
            # output_pydantic=str,
        )

    @crew
    def crew(self) -> Crew:
        """Creates the rumor identify Crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
