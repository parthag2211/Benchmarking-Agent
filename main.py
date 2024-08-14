import os
import json
from dotenv import load_dotenv
from crewai import Agent, Task, Crew
import requests
from langchain.tools import Tool

# Load environment variables
load_dotenv()

# Ensure the API keys are set
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("Please set the OPENAI_API_KEY environment variable in the .env file.")
if not os.getenv("SERPER_API_KEY"):
    raise ValueError("Please set the SERPER_API_KEY environment variable in the .env file.")

# Create a custom Serper search function
def search_serper(query):
    url = "https://google.serper.dev/search"
    payload = json.dumps({"q": query})
    headers = {
        'X-API-KEY': os.getenv('SERPER_API_KEY'),
        'Content-Type': 'application/json'
    }
    response = requests.request("POST", url, headers=headers, data=payload)
    return response.text

# Create a Serper tool using langchain's Tool
serper_tool = Tool(
    name='Serper Search',
    func=search_serper,
    description='Useful for searching information on the internet'
)

def get_user_input():
    companies = input("Please provide a list of companies (comma-separated): ").split(',')
    metrics = input("Please provide a list of metrics (comma-separated): ").split(',')
    return companies, metrics

def create_agents():
    user_input_agent = Agent(
        name='User Input Agent',
        role='User Input Specialist',
        goal='Ask the user for companies and benchmarking metrics.',
        backstory='You are skilled at gathering information from users and ensuring they provide the necessary details for the task.',
        verbose=True,
        allow_delegation=False
    )

    research_agent = Agent(
        name='Research Agent',
        role='Data Researcher',
        goal='Research the specified metrics for each company.',
        backstory='With an analytical mind, you excel at finding and compiling data from various online sources.',
        verbose=True,
        allow_delegation=True,
        tools=[serper_tool]
    )

    verification_agent = Agent(
        name='Verification Agent',
        role='Data Verifier',
        goal='Verify the accuracy of the researched data.',
        backstory='Your attention to detail ensures that all information is accurate and reliable.',
        verbose=True,
        allow_delegation=True,
        tools=[serper_tool]
    )

    return user_input_agent, research_agent, verification_agent

def create_tasks(user_input_agent, research_agent, verification_agent, companies, metrics):
    user_input_task = Task(
        description='Ask the user for a list of companies and the metrics to be benchmarked. Provide the user with a suggested list of metrics and get their approval.',
        agent=user_input_agent
    )

    research_task = Task(
        description=f'Research the specified metrics for each company: {companies} using various online resources and databases.',
        agent=research_agent
    )

    verification_task = Task(
        description=f'Verify the accuracy of the researched data by cross-checking with reliable sources for companies: {companies} and metrics: {metrics}.',
        agent=verification_agent
    )

    return [user_input_task, research_task, verification_task]

def main():
    print("Starting the CrewAI Benchmarking Tool...")

    companies, metrics = get_user_input()
    user_input_agent, research_agent, verification_agent = create_agents()
    tasks = create_tasks(user_input_agent, research_agent, verification_agent, companies, metrics)

    crew = Crew(
        agents=[user_input_agent, research_agent, verification_agent],
        tasks=tasks,
        verbose=2
    )

    print("Crew formed. Starting the process...")
    result = crew.kickoff()

    print("\nFinal Result:")
    print(result)

if __name__ == "__main__":
    main()