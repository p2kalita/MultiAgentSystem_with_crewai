import os

from crewai import LLM, Agent, Crew, Process, Task
from crewai_tools import FileReadTool, SerperDevTool
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")


# Configure Gemini LLM
gemini_llm = LLM(
    model="gemini/gemini-3.5-flash-lite", 
    temperature=0.2
)

# Create a sample local product spec file for the demo
with open("product_spec.txt", "w") as f:
    f.write(
        "Product: SupportIQ\n"
        "Core Feature: Automated ticket triage using Pandera data contracts and Prefect workflows.\n"
        "Target Audience: Mid-market SaaS support teams.\n"
        "Pricing: $499/month flat rate."    
    )

# Initialize Built-in Tools
file_tool = FileReadTool(file_path="product_spec.txt")
search_tool = SerperDevTool()

# Agent 1: Internal Product Analyst (reads local files)
product_analyst = Agent(
    role="Internal Product Analyst",
    goal="Extract accurate product features and pricing from internal documentation.",
    backstory="You thoroughly review internal product specs so the team never misstates capabilities.",
    tools=[file_tool],
    llm=gemini_llm,
    verbose=True)


# Agent 2: Market Intelligence Researcher (searches live web)
market_researcher = Agent(
    role="Market Intelligence Researcher",
    goal="Research current competitors and market trends using live web search.",
    backstory="You are an expert at finding up-to-date competitor positioning and pricing online.",
    tools=[search_tool],
    llm=gemini_llm,
    verbose=True)


# Agent 3: Sales Enablement Strategist (synthesizes both into a battlecard)
strategist = Agent(
    role="Sales Enablement Strategist",
    goal="Combine internal product strengths with market research into a concise sales battlecard.",
    backstory="You create sharp, actionable comparison sheets that help sales reps win deals.",
    llm=gemini_llm,
    verbose=True)


# Tasks
read_spec_task = Task(
    description="Read 'product_spec.txt' and summarize SupportIQ's core features, target audience, and pricing.",
    expected_output="A bulleted summary of SupportIQ's exact specs.",
    agent=product_analyst
)

research_competitors_task = Task(
    description="Search the web for top AI customer support ticket triage platforms in 2026 and note their key positioning.",
    expected_output="A list of 3 competitors with a 1-sentence positioning summary for each.",
    agent=market_researcher
)

battlecard_task = Task(
    description="Create a markdown Sales Battlecard comparing SupportIQ against the 3 researched competitors.",
    expected_output="A structured markdown comparison table plus 3 key talking points for sales reps.",
    agent=strategist,
    context=[read_spec_task, research_competitors_task]
)

# Assemble Crew
battlecard_crew = Crew(
    agents=[product_analyst, market_researcher, strategist],
    tasks=[read_spec_task, research_competitors_task, battlecard_task],
    process=Process.sequential,
    verbose=True)


if __name__ == "__main__":

    result = battlecard_crew.kickoff()
    print("\n--- SALES BATTLECARD ---\n")
    print(result)