from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.knowledge.source.text_file_knowledge_source import TextFileKnowledgeSource

textknowledge = TextFileKnowledgeSource(file_paths=["refute_example.txt"])


@CrewBase
class rumor_refute_crew:
    """rumor refute Crew"""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def rumor_refute_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["rumor_refute_agent"],
            verbose=True,
            llm="glm-4-flash",
        )

    @task
    def rumor_refute_task(self) -> Task:
        return Task(
            config=self.tasks_config["rumor_refute"],
        )

    @crew
    def crew(self) -> Crew:
        """Creates the rumor refute Crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
            # memory=True,
            knowledge_sources=[textknowledge],
        )
