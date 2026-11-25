import os

from crewai import LLM, Agent, Crew, Process, Task
from crewai.tools import tool
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Configure Gemini LLM
gemini_llm = LLM(
    model="gemini/gemini-3.5-flash-lite", 
    temperature=0.0
)

# 1. Define a Custom Tool using the @tool decorator
APPROVED_VENDORS = {
    "V-101": {"name": "CloudScale Hosting", "max_limit": 5000.0, "status": "ACTIVE"},
    "V-102": {"name": "QuickPrint Supplies", "max_limit": 500.0, "status": "SUSPENDED"},
}


@tool("Vendor Registry Lookup")
def check_vendor_registry(vendor_id: str) -> str:
    """
    Look up a vendor's approval status and maximum allowed invoice limit by vendor_id (e.g., 'V-101').
    Always call this tool before approving any invoice.
    """    
    vendor = APPROVED_VENDORS.get(vendor_id.strip())

    if not vendor:
        return f"Vendor {vendor_id} NOT FOUND in registry."
    
    return (
        f"Vendor: {vendor['name']} | Status: {vendor['status']} | "f"Max Approved Limit: ${vendor['max_limit']}"    
)


# 2. Define Pydantic Schema for Guaranteed Structured Output
class InvoiceAuditReport(BaseModel):    
    invoice_id: str = Field(description="The invoice identifier")
    vendor_id: str = Field(description="The vendor ID checked")
    is_approved: bool = Field(description="True only if vendor is ACTIVE and amount <= max_limit")
    flags: list[str] = Field(description="List of policy violations or 'None' if clean")
    summary: str = Field(description="One-sentence audit decision summary")


# 3. Define Agents
compliance_auditor = Agent(
    role="Procurement Compliance Auditor",
    goal="Verify vendor status and spending limits accurately using the Vendor Registry Lookup tool.",
    backstory="You are a meticulous financial auditor who strictly enforces company procurement policies.",
    tools=[check_vendor_registry],
    llm=gemini_llm,
    verbose=True)

finance_controller = Agent(
    role="Finance Controller",
    goal="Produce a final, machine-readable audit decision report based on compliance findings.",
    backstory="You sign off on audited invoices and format decisions cleanly for the ERP database.",
    llm=gemini_llm,
    verbose=True)

# 4. Define Tasks
verify_task = Task(
    description=(
        "Audit the following incoming invoice:\n""Invoice ID: {invoice_id}\n"
        "Vendor ID: {vendor_id}\n""Billed Amount: ${amount}\n"
        "Use the Vendor Registry Lookup tool to verify if the vendor is ACTIVE and within their limit."    
    ),
    expected_output="Detailed verification notes showing vendor status, limit, and whether the amount passes.",
    agent=compliance_auditor
)

decision_task = Task(
    description="Generate the final structured audit decision for Invoice {invoice_id}.",
    expected_output="A structured JSON audit report matching the InvoiceAuditReport schema.",
    agent=finance_controller,
    context=[verify_task],
    output_pydantic=InvoiceAuditReport  # Forces structured Pydantic output
)


# 5. Assemble Crew
audit_crew = Crew(
    agents=[compliance_auditor, finance_controller],
    tasks=[verify_task, decision_task],
    process=Process.sequential,
    verbose=True
)


if __name__ == "__main__":

    result = audit_crew.kickoff(
        inputs={
            "invoice_id": "INV-2026-889",
            "vendor_id": "V-102",
            "amount": 750.00        }
    )
    print("\n--- STRUCTURED PYDANTIC OUTPUT ---\n")
    # Access the validated Pydantic object directly via result.pydantic    
    print(result.pydantic.model_dump_json(indent=2))