from langchain.agents import create_agent
from langchain.tools import tool
from app.agents.common import MODEL, run_subagent
from app.lang_chain.tools import get_customers_tool, get_customer_cart_tool

# the specialist: it only gets the tools for its own job, so it cannot
# wander into product or policy questions.
customer_agent = create_agent(
    model=MODEL,
    tools=[get_customers_tool, get_customer_cart_tool],
    system_prompt=(
        "You handle customer account questions: who a customer is and what is in "
        "their cart. Use the tools, never guess data. If a tool returns an error "
        "or no data, say so plainly. Reply with the facts only, keep it short."
    ),
)


# the supervisor sees this wrapper as a normal tool. the docstring is how the
# supervisor decides WHEN to call the customer agent, so write it for the supervisor.
@tool
def ask_customer_agent(request: str) -> str:
    """Ask the customer specialist about a customer's details (name, email) or
    what is in their cart. Always include the customer id in the request."""
    return run_subagent(customer_agent, request)
