from langchain.agents import create_agent
from langchain.tools import tool
from app.agents.common import MODEL, run_subagent
from app.rag.search import search_policy


# the real policy search: embeds the question and finds the closest policy
# chunks in Pinecone (built in app/rag/).
@tool
def search_company_policy(query: str) -> str:
    """Search the company policy documents (returns, refunds, warranty, shipping)
    and return the most relevant policy text for the query."""
    try:
        return search_policy(query)
    except Exception as error:
        # Voyage or Pinecone can fail (network, quota). return the error as text
        # so the agent can say so, instead of crashing the whole run.
        return f"Error: {error}"


# the specialist for policy questions. its prompt is strict on purpose:
# answer only from what the tool returns, never from memory.
policy_agent = create_agent(
    model=MODEL,
    tools=[search_company_policy],
    system_prompt=(
        "You answer company policy questions (returns, refunds, warranty, "
        "shipping). Always call the policy search tool first. Answer ONLY from the "
        "text it returns, quote the relevant rule, and never use your own "
        "knowledge. If the tool has no policy text, say you cannot answer."
    ),
)


@tool
def ask_policy_agent(request: str) -> str:
    """Ask the policy specialist about company rules such as return windows,
    refund limits, warranty or shipping. Describe the situation in the request."""
    return run_subagent(policy_agent, request)
