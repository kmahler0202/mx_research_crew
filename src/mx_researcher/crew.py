from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
from dotenv import load_dotenv
load_dotenv()
from langchain_openai import ChatOpenAI
from crewai_tools import (
    SerperDevTool
)
from crewai.knowledge.source.text_file_knowledge_source import TextFileKnowledgeSource
import os

# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators



llm = ChatOpenAI(
    model=os.getenv('MODEL'),
    api_key=os.getenv('OPENAI_API_KEY'),
    temperature=0.7,
    max_completion_tokens=15000
)

search_tool = SerperDevTool()

specialist_knowledge = TextFileKnowledgeSource(
    file_paths=['graymatter_knowledge.txt']
)

strategy_knowledge = TextFileKnowledgeSource(
    file_paths=['strategies.txt']
)

@CrewBase
class MxResearcher():
    """MxResearcher crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    # Learn more about YAML configuration files here:
    # Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
    # Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended
    
    # If you would like to add tools to your agents, you can learn more about it here:
    # https://docs.crewai.com/concepts/agents#agent-tools

    agents_config = ('config/agents.yaml')
    tasks_config = ('config/tasks.yaml')

    @agent
    def planner(self) -> Agent:
        return Agent(
            config=self.agents_config['planner'], # type: ignore[index]
            tools=[search_tool],
            verbose=True,
            llm=llm,
            knowledge_sources=[specialist_knowledge]
        )

    
    @agent
    def researcher(self) -> Agent:
        return Agent(
            config=self.agents_config['researcher'], # type: ignore[index]
            tools=[search_tool],
            verbose=True,
            llm=llm,
            knowledge_sources=[strategy_knowledge]
        )

    @agent
    def synthesizer(self) -> Agent:
        return Agent(
            config=self.agents_config['synthesizer'], # type: ignore[index]
            verbose=True,
            llm=llm,
            knowledge_sources=[strategy_knowledge]
        )

    @agent
    def writer(self) -> Agent:
        return Agent(
            config=self.agents_config['writer'], # type: ignore[index]
            verbose=True,
            llm=llm    
        )

    # To learn more about structured task outputs,
    # task dependencies, and task callbacks, check out the documentation:
    # https://docs.crewai.com/concepts/tasks#overview-of-a-task

    @task
    def planning_task(self) -> Task:
        return Task(
            config=self.tasks_config['planning_task'], # type: ignore[index]
            output_file='planning_task.md'
        )

    @task
    def research_task(self) -> Task:
        return Task(
            config=self.tasks_config['research_task'], # type: ignore[index]
            output_file='research_task.md'
        )

    @task
    def synthesis_task(self) -> Task:
        return Task(
            config=self.tasks_config['synthesis_task'], # type: ignore[index]
            output_file='synthesis_task.md'
        )

    @task
    def reporting_task(self) -> Task:
        return Task(
            config=self.tasks_config['reporting_task'], # type: ignore[index]
            output_file='report.md'
        )

    @crew
    def crew(self) -> Crew:
        """Creates the MxResearcher crew"""
        # To learn how to add knowledge sources to your crew, check out the documentation:
        # https://docs.crewai.com/concepts/knowledge#what-is-knowledge

        return Crew(
            agents=self.agents, # Automatically created by the @agent decorator
            tasks=self.tasks, # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            # process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
        )
