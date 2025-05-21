from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.knowledge.source.text_file_knowledge_source import TextFileKnowledgeSource

textknowledge = TextFileKnowledgeSource(file_paths=["refute_example.txt"])

@CrewBase
class rumor_inoculation_crew:
    """rumor inoculation crew"""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def rumor_inoculation_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["rumor_inoculation_agent"],
            verbose=True,
            llm="glm-4-flash",
        )

    @task
    def rumor_inoculation_task(self) -> Task:
        return Task(
            config=self.tasks_config["rumor_inoculation"],
        )

    @crew
    def crew(self) -> Crew:
        """Creates the rumor inoculation crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
            # memory=True,
            knowledge_sources=[textknowledge],
        )
