import os

from crewai import LLM, Agent, Crew, Process, Task
from crewai.flow.flow import Flow, listen, router, start, or_
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

gemini_llm = LLM(
    model="gemini/gemini-3.5-flash-lite",
    temperature=0.2
)

# 1. Define Typed State shared across the entire Flow
class TicketTriageState(BaseModel):
    ticket_text: str = ""
    category: str = ""
    final_response: str = ""


# 2. Build Specialized Crew A: Technical Bug Escalation Crew
def build_bug_crew() -> Crew:
    engineer = Agent(
        role="Senior Support Engineer",
        goal="Reproduce software bugs and write clear GitHub issue summaries.",
        backstory="You translate customer bug reports into actionable reproduction steps for engineering.",
        llm=gemini_llm,
        verbose=True
    )

    task = Task(
        description="Analyze this bug report and create a structured engineering ticket:\n{ticket_text}",
        expected_output="A markdown bug summary with Severity, Component, and Reproduction Steps.",
        agent=engineer
    )

    return Crew(
        agents=[engineer],
        tasks=[task],
        process=Process.sequential
    )


# 3. Build Specialized Crew B: Sales & Upgrade Inquiry Crew
def build_sales_crew() -> Crew:
    account_exec = Agent(
        role="Enterprise Account Executive",
        goal="Respond warmly to pricing and plan upgrade inquiries to maximize conversion.",
        backstory="You help growing customers choose the right enterprise tier.",
        llm=gemini_llm,
        verbose=True
    )

    task = Task(
        description="Draft a tailored upgrade proposal response for this customer inquiry:\n{ticket_text}",
        expected_output="A friendly, persuasive response outlining next steps for a plan upgrade.",
        agent=account_exec
    )

    return Crew(
        agents=[account_exec],
        tasks=[task],
        process=Process.sequential
    )


# 4. Define the Event-Driven CrewAI Flow
class SupportTriageFlow(Flow[TicketTriageState]):

    @start()
    def ingest_ticket(self):
        print(f"\n[Flow Started] Ingested ticket: {self.state.ticket_text}")

    @router(ingest_ticket)
    def route_ticket(self):
        """
        Deterministic + LLM-assisted routing returning exact branch labels.
        """
        text = self.state.ticket_text.lower()

        if "crash" in text or "500 error" in text or "bug" in text:
            self.state.category = "BUG_REPORT"
            return "bug_route"
        else:
            self.state.category = "SALES_INQUIRY"
            return "sales_route"

    @listen("bug_route")
    def handle_bug(self):
        print("[Router -> Bug Crew] Spinning up Engineering Escalation Crew...")
        crew = build_bug_crew()
        result = crew.kickoff(inputs={"ticket_text": self.state.ticket_text})
        self.state.final_response = str(result)

    @listen("sales_route")
    def handle_sales(self):
        print("[Router -> Sales Crew] Spinning up Enterprise Sales Crew...")
        crew = build_sales_crew()
        result = crew.kickoff(inputs={"ticket_text": self.state.ticket_text})
        self.state.final_response = str(result)

    @listen(or_("handle_bug", "handle_sales"))
    def finalize_output(self):
        print(f"\n[Flow Completed] Category: {self.state.category}")
        return self.state.final_response


if __name__ == "__main__":
    # Test 1: Trigger the Bug Route
    flow_bug = SupportTriageFlow()

    output_1 = flow_bug.kickoff(
        inputs={
            "ticket_text": "Our webhook endpoint throws a 500 error and crashes whenever payload > 5MB."
        }
    )

    print("\n--- RESULT 1 (BUG ROUTE) ---\n", output_1)

    # Test 2: Trigger the Sales Route
    flow_sales = SupportTriageFlow()

    output_2 = flow_sales.kickoff(
        inputs={
            "ticket_text": "We are expanding to 150 support agents next month and want annual enterprise pricing."
        }
    )

    print("\n--- RESULT 2 (SALES ROUTE) ---\n", output_2)