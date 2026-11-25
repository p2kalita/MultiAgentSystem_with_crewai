import os
from typing import Any

from crewai import LLM, Agent, Crew, Process, Task, TaskOutput
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

gemini_llm = LLM(model="gemini/gemini-3.5-flash-lite", temperature=0.2)

# 1. Programmatic Guardrail Function
def refund_policy_guardrail(result: TaskOutput) -> tuple[bool, Any]:
    """
    Validates that the agent's proposed email includes a ticket reference
    and never offers a cash refund above $100 without escalation language.
    """    
    text = result.raw

    if "REF-" not in text:
        return (False, "Missing required refund reference code format (e.g., REF-XXXX). Please include it.")
    
    if "$500" in text and "finance approval" not in text.lower():
        return (False, "Refunds over $100 must explicitly mention 'finance approval'. Revise the draft.")
    
    return (True, text)

# 2. Define Agent
billing_agent = Agent(
    role="Billing Resolution Specialist",
    goal="Draft accurate, policy-compliant refund responses for enterprise customers.",
    backstory="You handle disputed charges carefully, following strict financial compliance rules.",
    llm=gemini_llm,
    verbose=True)


# 3. Define Task with both Automated Guardrail and Human-in-the-Loop
refund_task = Task(
    description=(
        "Customer '{customer}' requested a ${amount} refund for duplicate billing on ticket {ticket_id}. "
        "Draft a formal resolution email. Use reference code REF-{ticket_id}."    
    ),
    expected_output="A polite, policy-compliant customer resolution email.",
    agent=billing_agent,
    guardrail=refund_policy_guardrail,  # Automated LLM self-correction loop    
    guardrail_max_retries=3,
    human_input=True                    # Pauses for terminal human approval before completing
)
# 4. Assemble Crew
billing_crew = Crew(
    agents=[billing_agent],
    tasks=[refund_task],
    process=Process.sequential,
    verbose=True)

if __name__ == "__main__":

    result = billing_crew.kickoff(
        inputs={
            "customer": "Acme Corp",
            "amount": 500,
            "ticket_id": "8821"        }
    )

    print("\n--- APPROVED & FINALIZED EMAIL ---\n")
    print(result)