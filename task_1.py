import os

from crewai import LLM, Agent, Crew, Process, Task
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Configure Gemini LLM (can also use different temperatures per agent)
planner_llm = LLM(model="gemini/gemini-3.5-flash-lite", temperature=0.2)
writer_llm = LLM(model="gemini/gemini-3.5-flash-lite", temperature=0.7)


# Agent 1: Planner / Architect
architect = Agent(
    role="Tech Curriculum Architect",
    goal="Design clear, logical, beginner-friendly outlines for technical topics.",
    backstory=(
        "You break down complex software engineering concepts into step-by-step ""learning modules with practical real-world analogies."    
    ),
    llm=planner_llm,
    verbose=True)


# Agent 2: Writer
writer = Agent(
    role="Senior Technical Writer",
    goal="Write engaging, easy-to-read technical blog posts based on structured outlines.",
    backstory=(
        "You write crisp, jargon-free engineering articles filled with concrete ""examples that developers love reading."    
    ),
    llm=writer_llm,
    verbose=True)


# Task 1: Create the outline
outline_task = Task(
    description="Create a 4-section blog outline explaining '{topic}' to junior developers.",
    expected_output="A structured markdown outline with section titles and 2 bullet points per section.",
    agent=architect
)


# Task 2: Write the post using Task 1's output automatically
write_task = Task(
    description=(
        "Using the outline provided by the Tech Curriculum Architect, write a ""complete, engaging 400-word blog post on '{topic}'. Include one concrete analogy."    
    ),
    expected_output="A polished, ready-to-publish markdown blog post.",
    agent=writer,
    context=[outline_task]  # Explicitly passes outline_task output to this task
)

# Assemble the Sequential Crew
content_crew = Crew(
    agents=[architect, writer],
    tasks=[outline_task, write_task],
    process=Process.sequential,
    verbose=True
)

if __name__ == "__main__":

    result = content_crew.kickoff(inputs={"topic": "Idempotency in Data Pipelines"})

    print("\n--- PUBLISHED ARTICLE ---\n")
    print(result)