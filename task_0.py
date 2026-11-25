import os

from crewai import LLM, Agent, Crew, Process, Task
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# 1. Configure the Gemini LLM
gemini_llm = LLM(
    model="gemini/gemini-3.5-flash-lite",
    temperature=0.3)

# 2. Define the Agent
support_specialist = Agent(
    role="Customer Support Specialist",
    goal="Turn rough internal notes into polite, clear, and helpful customer emails.",
    backstory=(
        "You are a seasoned customer experience lead known for turning technical"
        "explanations into warm, reassuring responses that build customer trust."    
    ),
    llm=gemini_llm,
    verbose=True
)

# 3. Define the Task
draft_email_task = Task(
    description=(
            "Rewrite the following internal engineer note into a customer-facing email:\n"
            "'{raw_note}'"    
        ),
    expected_output="A polite, concise 3-paragraph email ready to send to the customer.",
    agent=support_specialist
)

# 4. Assemble the Crew
crew = Crew(
    agents=[support_specialist],
    tasks=[draft_email_task],
    process=Process.sequential,
    verbose=True
)


if __name__ == "__main__":

    result = crew.kickoff(
    inputs={
            "raw_note": "DB connection pool maxed out at 02:15 UTC. Restarted worker nodes."
            "Data wasn't lost, just delayed by 20 mins. Fixed now."       
        }
    )

print("\n--- FINAL OUTPUT ---\n")
print(result)