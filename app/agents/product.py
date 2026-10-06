from langchain.agents import create_agent
from langchain.tools import tool
from app.agents.common import MODEL, run_subagent
from app.lang_chain.tools import (
    get_product_tool,
    search_products_tool,
    get_products_by_category_tool,
)

# the specialist for the product catalog, with only the product tools
product_agent = create_agent(
    model=MODEL,
    tools=[get_product_tool, search_products_tool, get_products_by_category_tool],
    system_prompt=(
        "You handle product catalog questions: finding products, comparing prices, "
        "and recommending cheaper options. Use the tools, never guess data. If a "
        "tool returns an error or no data, say so plainly. Reply with the facts "
        "only (title, price, category), keep it short."
    ),
)


@tool
def ask_product_agent(request: str) -> str:
    """Ask the product specialist to find, search, compare or recommend products
    in the catalog. Include any price limit or category in the request."""
    return run_subagent(product_agent, request)
