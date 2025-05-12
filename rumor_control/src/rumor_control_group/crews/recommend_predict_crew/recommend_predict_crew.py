from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task


@CrewBase
class recommend_predict_crew:
    """recommend_predict_crew"""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def hr_evaluation_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["hr_evaluation_agent"],
            verbose=True,
        )

    @task
    def evaluate_candidate_task(self) -> Task:
        return Task(
            config=self.tasks_config["evaluate_candidate"],
        )

    @crew
    def crew(self) -> Crew:
        """Creates the recommend_predict_crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
