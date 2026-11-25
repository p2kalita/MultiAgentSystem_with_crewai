import os

from crewai import LLM, Agent, Crew, Process, Task
from crewai.tools import tool
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Configure Gemini LLMs
worker_llm = LLM(model="gemini/gemini-3.5-flash-lite", temperature=0.1)
manager_llm = LLM(model="gemini/gemini-3.5-flash-lite", temperature=0.2)

# Custom Diagnostic Tools
@tool("Check Kubernetes Pods")
def check_k8s_pods(service_name: str) -> str:
    """Inspect pod health, CPU/memory usage, and restart counts for a given service."""

    return (
        f"Service '{service_name}': 4/4 pods Running. Memory at 42%. "
        "Zero OOMKills. API latency spiked to 4,200ms at 02:14 UTC."    
    )

@tool("Check Postgres Locks")
def check_db_locks(database_name: str) -> str:
    """Check active locks, slow queries, and connection pool saturation in PostgreSQL."""

    return (
        f"Database '{database_name}': 98/100 connections active. "
        "Query 'ALTER TABLE tickets ADD COLUMN...' holding ACCESS EXCLUSIVE lock for 185s."    
    )

# Specialist Agents (Note: allow_delegation=False keeps specialists focused on their own tools)
sre_agent = Agent(
    role="Infrastructure SRE",
    goal="Investigate compute, container health, and network latency metrics.",
    backstory="You are a site reliability engineer who diagnoses Kubernetes and application-layer bottlenecks.",
    tools=[check_k8s_pods],
    llm=worker_llm,
    allow_delegation=False,
    verbose=True)

dba_agent = Agent(
    role="Database Reliability Engineer",
    goal="Diagnose database locks, slow queries, and connection pool exhaustion.",
    backstory="You are a PostgreSQL internals expert who tracks down blocking transactions and schema lock contention.",
    tools=[check_db_locks],
    llm=worker_llm,
    allow_delegation=False,
    verbose=True)


# High-level Task (The Manager decides who does what and in what order)
rca_task = Task(
    description=(
        "Investigate the production outage alert: '{alert_message}'."
        "Coordinate with the Infrastructure SRE and Database Reliability Engineer to find"
        "the exact root cause and produce a concise Root Cause Analysis (RCA) report."    
    ),
    expected_output=(
        "A structured RCA report containing: Incident Summary, Root Cause."
        "Evidence from Logs/Metrics, and Immediate Remediation Steps."    
    )
)

# Assemble Hierarchical Crew
incident_crew = Crew(
    agents=[sre_agent, dba_agent],
    tasks=[rca_task],
    process=Process.hierarchical,
    manager_llm=manager_llm,  # CrewAI auto-creates a Manager Agent using this LLM   
    verbose=True
)


if __name__ == "__main__":
    result = incident_crew.kickoff(
        inputs={
            "alert_message": "SupportIQ API p99 latency exceeded 4s on 'tickets-api' connected to 'supportiq_prod' DB."        }
    )
    print("\n--- HIERARCHICAL RCA REPORT ---\n")
    print(result)